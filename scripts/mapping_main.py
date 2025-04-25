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

# Função para simular a trajetória das granadas (simplificada)
def simulate_grenade_trajectory(start_x, start_y, start_z, velocity, angle, gravity=-9.8, duration=3.0):
    times = np.linspace(0, duration, num=30)  # 30 pontos ao longo do tempo
    x_vals = start_x + velocity * np.cos(angle) * times
    y_vals = start_y + velocity * np.sin(angle) * times
    z_vals = start_z + velocity * times - 0.5 * gravity * times**2
    return list(zip(x_vals, y_vals, z_vals))

# Função para determinar o tipo de granada e sua área de efeito
def get_grenade_effect(grenade_type, x, y, z, tick):
    # Mapeamento dos tipos de granada reais para tipos simplificados + raio
    grenade_map = {
        "CFlashbang": ("flashbang", 5.0),
        "CFlashbangProjectile": ("flashbang", 5.0),
        "CHEGrenade": ("hegrenade", 3.0),
        "CHEGrenadeProjectile": ("hegrenade", 3.0),
        "CMolotovProjectile": ("molotov", 3.0),
        "CIncendiaryGrenade": ("molotov", 3.0),
        "CMolotovGrenade": ("molotov", 3.0),
        "CSmokeGrenade": ("smoke", 7.0),
        "CSmokeGrenadeProjectile": ("smoke", 7.0)
    }

    if grenade_type in grenade_map:
        simplified_type, radius = grenade_map[grenade_type]
        return {
            "type": simplified_type,
            "position": (x, y, z),
            "radius": radius,
            "tick": tick
        }

    return None

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

        # Verifique se há uma granada lançada neste tick
        for grenade in dem.grenades.iter_rows(named=True):  # Iterando sobre as linhas do DataFrame
            # Aqui você já tem os dados de cada granada como dicionário
            if grenade['tick'] == tick:
                # Verifique o tipo de granada (flashbang, HE, molotov)
                grenade_effect = get_grenade_effect(grenade['grenade_type'], grenade['X'], grenade['Y'], grenade['Z'], tick)
                if grenade_effect:
                    grenades.append(grenade_effect)

    frames.append({"points": points, "point_settings": point_settings, "grenades": grenades})

print("Finished processing frames. Creating gif...")
gif("de_inferno", frames, "de_inferno.gif", duration=100)