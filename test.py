import geopandas as gpd

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


# Filtrare Asia
asia = world[world["continent"] == "Asia"].copy()

# Repară geometriile înainte de salvare
asia["geometry"] = asia["geometry"].buffer(0)

# Salvare fișier valid GeoJSON
asia.to_file("asia_valid.geojson", driver="GeoJSON")