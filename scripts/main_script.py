# Imports
import pandas as pd
import numpy as np
import os
import zipfile
from awpy import Demo

# Setting parameters for the CS Demo Manager Extractions
folder_path = r'C:\Users\bayli\Documents\Git Projects\CS2\xlsx_exports'
sheet_data = {}
df_all_first_kills = pd.DataFrame()

# CS Demo Manager Extrations
# Gets data from all files in the folder
# Gets a separate loop for the 'kills' tab in order to get the opening duels in a separate 'kills' dataframe
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        file_path = os.path.join(folder_path, file_name)
        excel_data = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, df in excel_data.items():
            if sheet_name == 'Kills':
                df_Kills = df
                first_kills = df_Kills.sort_values(by=['round_number', 'tick'])
                first_kills = first_kills.groupby('round_number').first().reset_index()
                df_all_first_kills = pd.concat([df_all_first_kills, first_kills], ignore_index=True)
            if sheet_name in sheet_data:
                sheet_data[sheet_name] = pd.concat([sheet_data[sheet_name], df], ignore_index=True)
            else:
                sheet_data[sheet_name] = df

# Creates the dataframes based on each tab
df_General = sheet_data['General']
df_Rounds = sheet_data['Rounds']
df_Players = sheet_data['Players']
df_Kills = sheet_data['Kills']
df_Weapons = sheet_data['Weapons']
df_PFM = sheet_data['Players Flashbang matrix']
df_Clutches = sheet_data['Clutches']

# Drops color (useless)
df_Players = df_Players.drop('color', axis=1)

# Creates rounds won columns
team_rounds_won = df_Rounds.groupby('winner_name').agg(
    total_rounds_won=('winner_name', 'size'),
    t_rounds_won=('winner_side', lambda x: (x == 2).sum()),
    ct_rounds_won=('winner_side', lambda x: (x == 3).sum())
).reset_index()

team_rounds_won.columns = ['team_name', 'total_rounds_won','t_rounds_won', 'ct_rounds_won']

# Creates the player tabler with all the kills raw data grouped for every match in a single table
player_kills = df_Kills.groupby('killer_steam_id').agg(
    kills=('killer_steam_id', 'size'),
    headshots=('is_headshot', 'sum'),
    wallbang_kills=('penetrated_objects', 'sum'),
    assisted_flashes=('is_assisted_flash', 'sum'),
    trade_kills=('is_trade_kill', 'sum'),
    trade_deaths=('is_trade_death', 'sum'),
    no_scope=('is_no_scope', 'sum'),
    through_smoke=('is_through_smoke', 'sum'),
    airbone_kills=('is_killer_airbone', 'sum'),
    airbone_victim_kills=('is_victim_airbone', 'sum'),
    blind_kills=('is_killer_blinded', 'sum'),
    victim_blind_kills=('is_victim_blinded', 'sum'),
).reset_index()
player_kills.rename(columns={'killer_steam_id': 'steam_id'}, inplace=True)

# Opening Duels
# Creates the player tabler with all the kills raw data grouped for every match in a single table
player_first_kills = df_all_first_kills.groupby('killer_steam_id').agg(
    kills=('killer_steam_id', 'size'),
    first_deaths=('victim_steam_id', 'size'),
    CT_first_kills=('killer_side', lambda x: len(x[x == 3])),
    T_first_kills=('killer_side', lambda x: len(x[x == 2])),
    CT_first_deaths=('victim_side', lambda x: len(x[x == 3])),
    T_first_deaths=('victim_side', lambda x: len(x[x == 2])),
).reset_index()
player_first_kills.rename(columns={'killer_steam_id': 'steam_id'}, inplace=True)

# Creates new raw data from the 'df_Players'
df_Players_1 = df_Players.groupby('steam_id', as_index=False).agg({'name': 'first',
    'team_name': 'first',
    'kill_count': 'sum', 
    'assist_count': 'sum',
    'kd':'mean',
    'mvp':'sum',
    'HLTV':'mean',
    'HLTV 2.0':'mean',
    'kast':'mean',
    'death_count': 'sum',
    'headshot_count': 'sum',
    'first_kill_count': 'sum',
    'first_death_count': 'sum',
    'bomb_defused_count': 'sum',
    'bomb_planted_count': 'sum',
    '1v1': 'sum',
    '1v2': 'sum',
    '1v3': 'sum',
    '1v4': 'sum',
    '1v5': 'sum',
    '1v1_won': 'sum',
    '1v2_won': 'sum',
    '1v3_won': 'sum',
    '1v4_won': 'sum',
    '1v5_won': 'sum',
    '1v2_lost': 'sum',
    '1v3_lost': 'sum',
    '1v4_lost': 'sum',
    '1v5_lost': 'sum',
})

# Merges all the current data in a single dataframe
df_Players_1 = df_Players_1.merge(team_rounds_won, on='team_name', how='left')
df_Players_1 = df_Players_1.merge(player_kills, on='steam_id', how='left')
df_Players_1 = df_Players_1.merge(player_first_kills, on='steam_id', how='left')

# Sets Parameters for the AWPY Extractions
folder_path = r'C:\Users\bayli\Documents\Git Projects\test_demos'

df_flashes = pd.DataFrame()
df_he = pd.DataFrame()
df_infernos = pd.DataFrame()
df_smoke = pd.DataFrame()

# AWPY Extractions
# Gets all the grenades data from all the .dem files in the folder 
for file_name in os.listdir(folder_path):
    if file_name.endswith('.dem'):

        file_path = os.path.join(folder_path, file_name)
        dem = Demo(file_path)

        this_file_flashes = dem.events['flashbang_detonate']
        this_file_he = dem.events['hegrenade_detonate']
        this_file_infernos = dem.events['inferno_startburn']
        this_file_smoke = dem.events['smokegrenade_detonate']

        df_flashes = pd.concat([df_flashes,this_file_flashes], ignore_index=True)
        df_he = pd.concat([df_he,this_file_he], ignore_index=True)
        df_infernos = pd.concat([df_infernos,this_file_infernos], ignore_index=True)
        df_smoke = pd.concat([df_smoke,this_file_smoke], ignore_index=True)

# Group all the grenades data in separated dataframes
df_all_flashes = df_flashes.groupby('user_steamid').agg(
    flahes_thrown=('user_steamid', 'size'),
    CT_flahes_thrown=('user_team_name', lambda x: (x == 'CT').sum()),
    T_flahes_thrown=('user_team_name', lambda x: (x == 'TERRORIST').sum()),
    flahes_thrown_in_pistol_round=('round', lambda x: ((x == 1) | (x == 13)).sum()),
).reset_index()
df_all_flashes.rename(columns={'user_steamid': 'steam_id'}, inplace=True)
df_all_flashes['steam_id'] = df_all_flashes['steam_id'].astype('int64')

df_all_he = df_he.groupby('user_steamid').agg(
    he_thrown=('user_steamid', 'size'),
    CT_he_thrown=('user_team_name', lambda x: (x == 'CT').sum()),
    T_he_thrown=('user_team_name', lambda x: (x == 'TERRORIST').sum()),
    he_thrown_in_pistol_round=('round', lambda x: ((x == 1) | (x == 13)).sum())
).reset_index()
df_all_he.rename(columns={'user_steamid': 'steam_id'}, inplace=True)
df_all_he['steam_id'] = df_all_he['steam_id'].astype('int64')

df_all_infernos = df_infernos.groupby('user_steamid').agg(
    infernos_thrown=('user_steamid', 'size'),
    CT_infernos_thrown=('user_team_name', lambda x: (x == 'CT').sum()),
    T_infernos_thrown=('user_team_name', lambda x: (x == 'TERRORIST').sum()),
    infernos_thrown_in_pistol_round=('round', lambda x: ((x == 1) | (x == 13)).sum())
).reset_index()
df_all_infernos.rename(columns={'user_steamid': 'steam_id'}, inplace=True)
df_all_infernos['steam_id'] = df_all_infernos['steam_id'].astype('int64')

df_all_smokes = df_smoke.groupby('user_steamid').agg(
    smokes_thrown=('user_steamid', 'size'),
    CT_smokes_thrown=('user_team_name', lambda x: (x == 'CT').sum()),
    T_smokes_thrown=('user_team_name', lambda x: (x == 'TERRORIST').sum()),
    smokes_thrown_in_pistol_round=('round', lambda x: ((x == 1) | (x == 13)).sum())
).reset_index()
df_all_smokes.rename(columns={'user_steamid': 'steam_id'}, inplace=True)
df_all_smokes['steam_id'] = df_all_smokes['steam_id'].astype('int64')

# Merges all the current data in a single dataframe
df_Players_1 = df_Players_1.merge(df_all_flashes, on='steam_id', how='left')
df_Players_1 = df_Players_1.merge(df_all_he, on='steam_id', how='left')
df_Players_1 = df_Players_1.merge(df_all_infernos, on='steam_id', how='left')
df_Players_1 = df_Players_1.merge(df_all_smokes, on='steam_id', how='left')

df_Players_1['util_in_pistol_round'] = df_Players_1['flahes_thrown_in_pistol_round'] + df_Players_1['he_thrown_in_pistol_round'] + df_Players_1['infernos_thrown_in_pistol_round'] +  df_Players_1['smokes_thrown_in_pistol_round']
df_Players_1['total_util_thrown'] = df_Players_1['flahes_thrown'] + df_Players_1['he_thrown'] + df_Players_1['infernos_thrown'] +  df_Players_1['smokes_thrown']

df_Players_1.to_csv('players.csv')
