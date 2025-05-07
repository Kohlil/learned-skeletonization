import os
import random
import numpy as np
import geopandas as gpd
import osmnx as ox
from osmnx.projection import project_gdf
import rasterio
from rasterio.features import rasterize
from shapely.geometry import LineString, Point, mapping
from PIL import Image, ImageDraw
import json
from tqdm import tqdm

# Keep parameters bundled

CONFIGS = {
    "train": {
        "place": "Beavercreek, Ohio, USA",
        "image_size": (256, 256),
        "n_samples": 1000,
        "network_type": "drive",
        "data_dir": "data/train/thinning",
    },
    "test": {
        "place": "Middletown, Ohio, USA",
        "image_size": (256, 256),
        "n_samples": 500,
        "network_type": "drive",
        "data_dir": "data/test/thinning",
    },
}

for c in CONFIGS:
    CONFIGS[c]["name"] = c


# Default thickness by road type (in meters)
# formatted as (lower, upper)
DEFAULT_THICKNESS = {
    "motorway": (20, 35),
    "trunk": (15, 30),
    "primary": (10, 25),
    "secondary": (7, 15),
    "tertiary": (5, 10),
    "unclassified": (3, 7),
    "residential": (4, 8),
    "living_street": (3, 6),
    "service": (2.5, 6),
    "pedestrian": (2, 8),
    "track": (2, 4),
    "footway": (1, 3),
    "cycleway": (1.5, 4),
    "bridleway": (1.5, 3),
    "steps": (1, 3),
    "path": (0.5, 2.5),
    "motorway_link": (7, 15),
    "trunk_link": (6, 12),
    "primary_link": (6, 12),
    "secondary_link": (5, 10),
    "tertiary_link": (4, 8),
    "bus_guideway": (6, 8),
    "raceway": (10, 20),
    "road": (3, 8),
    "busway": (6, 8),
    "corridor": (1, 3),
    "via_ferrata": (0.5, 1.5),
    "sidewalk": (1, 3),
    "crossing": (2, 6),
    "traffic_island": (1, 3),
}


def get_default_thickness(row):
    tag = row.get("highway", "residential")
    if isinstance(tag, list):
        tag = tag[0]  # pick the first tag if it's a list
    lower, upper = DEFAULT_THICKNESS.get(tag, (2, 4))
    return random.uniform(lower, upper)


def download_osm_data(config, cache_dir="cache"):
    config_name = config.get("name", "default")
    place = config["place"]
    network_type = config.get("network_type", "all")

    config_cache_dir = os.path.join(cache_dir, config_name)
    os.makedirs(config_cache_dir, exist_ok=True)

    edges_fp = os.path.join(config_cache_dir, "edges.gpkg")
    nodes_fp = os.path.join(config_cache_dir, "nodes.gpkg")

    if os.path.exists(edges_fp) and os.path.exists(nodes_fp):
        print(f"Loading cached OSM data for {place}...")
        gdf_edges = gpd.read_file(edges_fp)
        gdf_nodes = gpd.read_file(nodes_fp)
    else:
        print(f"Fetching roads from {place}...")
        G = ox.graph_from_place(place, network_type=network_type)
        print("Converting to dataframes...")
        gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        gdf_nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)

        print("Projecting to UTM ...")
        gdf_edges = project_gdf(gdf_edges)  # Automatically choosed a UTM zone
        gdf_nodes = project_gdf(gdf_nodes)

        print(f"Fetched {len(gdf_edges)} edges and {len(gdf_nodes)} nodes.")
        gdf_edges.to_file(edges_fp, driver="GPKG")
        gdf_nodes.to_file(nodes_fp, driver="GPKG")

    return gdf_edges, gdf_nodes


def pick_random_intersections(nodes, n=500):
    return nodes.sample(n)


def rasterize_roads(roads, bounds, size, thickness_fn):
    """
    Rasterize roads into a binary mask using rasterio.
    Thickness is applied by buffering each LineString in meters.
    """
    # Create transform (affine mapping from pixel coords to geographic coords)
    transform = rasterio.transform.from_bounds(*bounds, width=size[0], height=size[1])

    # Buffer each line to simulate thickness and generate (geometry, value) tuples
    shapes = []
    for _, row in roads.iterrows():
        geom = row.geometry
        if geom is None or not isinstance(geom, LineString):
            continue
        thickness = thickness_fn(row) / 2.0  # buffer radius in meters
        if thickness > 0:
            buffered = geom.buffer(thickness)
            shapes.append((buffered, 255))  # Burn value = 1 for road
        else:
            shapes.append((geom, 255))

    mask = rasterize(
        shapes=shapes, out_shape=size, transform=transform, fill=0, dtype=np.uint8
    )
    return mask


def generate_samples(
    gdf_edges, gdf_nodes, n_samples=500, out_dir="dataset", image_size=(256, 256)
):
    os.makedirs(out_dir, exist_ok=True)

    selected_nodes = pick_random_intersections(gdf_nodes, n=n_samples)

    for i, (_, node) in enumerate(tqdm(selected_nodes.iterrows(), total=n_samples)):
        pt = node.geometry
        buffer_m = 128  # meters around the point
        bounds = (pt.x - buffer_m, pt.y - buffer_m, pt.x + buffer_m, pt.y + buffer_m)
        clip = gdf_edges.cx[bounds[0] : bounds[2], bounds[1] : bounds[3]].copy()
        if clip.empty:
            continue

        image_array = rasterize_roads(clip, bounds, image_size, get_default_thickness)
        target_array = rasterize_roads(clip, bounds, image_size, lambda row: 0.0)

        # Save image
        image_path = os.path.join(out_dir, f"image_{i:05d}.png")
        Image.fromarray(image_array).save(image_path)

        # Save target (same as image for this example)
        target_path = os.path.join(out_dir, f"target_{i:05d}.png")
        Image.fromarray(target_array).save(target_path)

        # Save geojson
        target_geojson_path = os.path.join(out_dir, f"target_{i:05d}.geojson")
        features = [
            {
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {"highway": row.get("highway", "unknown")},
            }
            for _, row in clip.iterrows()
            if isinstance((geom := row.geometry), LineString)
        ]

        geojson = {"type": "FeatureCollection", "features": features}
        with open(target_geojson_path, "w") as f:
            json.dump(geojson, f)


if __name__ == "__main__":
    req_data = ["train", "test"]
    for config_key in req_data:
        config = CONFIGS[config_key]
        edges, nodes = download_osm_data(config)
        generate_samples(
            edges,
            nodes,
            n_samples=config["n_samples"],
            out_dir=config.setdefault("data_dir", "./data/thinning"),
        )
