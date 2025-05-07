# Imports
import pandas as pd
import polars as pl
import numpy as np
import os
from awpy import Demo
from awpy.plot import gif, PLOT_SETTINGS
from tqdm import tqdm

folder_path = r'C:\Users\bayli\Documents\Git Projects\test_demos'

df_flashes = pd.DataFrame()
df_he = pd.DataFrame()
df_infernos = pd.DataFrame()
df_smoke = pd.DataFrame()

file_path = r'C:\Users\bayli\Documents\Git Projects\test_demos\natus-vincere-vs-faze-m1-inferno.dem'
dem = Demo(file_path)
dem.parse(player_props=["X", "Y", "Z", "health", "armor_value", "pitch", "yaw", "side", "name"])

frames = []

for tick in tqdm(dem.ticks.filter(pl.col("round_num") == 1)["tick"].unique().to_list()[::32]):
    frame_df = dem.ticks.filter(pl.col("tick") == tick)
    frame_df = frame_df[
        ["X", "Y", "Z", "health", "armor", "pitch", "yaw", "side", "name"]
    ]

    points = []
    point_settings = []
    grenades = []

    for row in frame_df.iter_rows(named=True):
        points.append((row["X"], row["Y"], row["Z"]))

        # Determine team and corresponding settings
        settings = PLOT_SETTINGS[row["side"]].copy()

        # Add additional settings
        settings.update(
            {
                "hp": row["health"],
                "armor": row["armor"],
                "direction": (row["pitch"], row["yaw"]),
                "label": row["name"],
            }
        )

        point_settings.append(settings)

    frames.append({"points": points, "point_settings": point_settings})

print("Finished processing frames. Creating gif...")
gif("de_inferno", frames, "de_inferno.gif", duration=100)