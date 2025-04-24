# Imports
import pandas as pd
import numpy as np
import polars as pl
import os
import awpy
from demoparser2 import DemoParser
from awpy import Demo
from pathlib import Path
from awpy.stats import adr
from awpy.stats import kast
from awpy.stats import rating
from awpy.stats import calculate_trades

folder_path = r'C:\Users\bayli\Documents\CS Demos\All_IEM_Katowice_2025\IEM_Katowice_2025'
#folder_path = r'C:\Users\bayli\Documents\Git Projects\test_demos'

# Creating DataFrames
df_flashes = pd.DataFrame()
df_he = pd.DataFrame()
df_infernos = pd.DataFrame()
df_smoke = pd.DataFrame()
df_kills = pd.DataFrame()
df_rounds = pd.DataFrame()
df_all_first_kills = pd.DataFrame()
df_adr = pd.DataFrame()
df_kast = pd.DataFrame()
df_util_dmg = pd.DataFrame()
team_rounds_won = pd.DataFrame()
players_id = pd.DataFrame()
df_matches = pd.DataFrame(columns=['file_id', 'file_name'])
i = 1
file_id_counter = 1

def add_round_winners(ticks_df, rounds_df):
    ticks_df = ticks_df.to_pandas()
    rounds_df = rounds_df.to_pandas()
    
    # Changes the first round losing steak to 0
    # This is necessary because the first round is not a losing streak
    ticks_df.loc[ticks_df['round_num'] == 1, ['ct_losing_streak', 't_losing_streak']] = 0

    # Makes sure the columns exists
    rounds_df['CT_team_clan_name'] = None
    rounds_df['T_team_clan_name'] = None
    rounds_df['winner_clan_name'] = None
    rounds_df['CT_team_current_equip_value'] = None
    rounds_df['T_team_current_equip_value'] = None
    rounds_df['ct_losing_streak'] = None
    rounds_df['t_losing_streak'] = None

    for idx, row in rounds_df.iterrows():
        freeze_end_tick = row['freeze_end']
        winner = row['winner']

        # Takes all corresponding entries
        first_tick_df = ticks_df[ticks_df['tick'] == freeze_end_tick]

        # Takes the name for every team
        try:
            CT_team = first_tick_df[first_tick_df['side'] == 'ct']['team_clan_name'].iloc[0]
        except IndexError:
            CT_team = None
        
        try:
            T_team = first_tick_df[first_tick_df['side'] == 't']['team_clan_name'].iloc[0]
        except IndexError:
            T_team = None

        # Takes the current equip value for every team
        try:
            CT_team_current_equip_value = first_tick_df[first_tick_df['side'] == 'ct']['current_equip_value'].sum()
        except KeyError:
            CT_team_current_equip_value = None

        try:
            T_team_current_equip_value = first_tick_df[first_tick_df['side'] == 't']['current_equip_value'].sum()
        except KeyError:
            T_team_current_equip_value = None

        # Takes the losing streak for every team
        try:
            ct_losing_streak = first_tick_df[first_tick_df['side'] == 'ct']['ct_losing_streak'].iloc[0]
        except IndexError:
            ct_losing_streak = None

        try:
            t_losing_streak = first_tick_df[first_tick_df['side'] == 't']['t_losing_streak'].iloc[0]
        except IndexError:
            t_losing_streak = None

        if winner == 'ct':
            winner_clan = CT_team
        elif winner in ['t', 'TERRORIST']:
            winner_clan = T_team
        else:
            winner_clan = None
            print(f"[!] Round {idx} - winner error: '{winner}'")
            
        # Fill Columns in the DataFrame
        rounds_df.at[idx, 'CT_team_clan_name'] = CT_team
        rounds_df.at[idx, 'T_team_clan_name'] = T_team
        rounds_df.at[idx, 'winner_clan_name'] = winner_clan
        rounds_df.at[idx, 'CT_team_current_equip_value'] = CT_team_current_equip_value
        rounds_df.at[idx, 'T_team_current_equip_value'] = T_team_current_equip_value
        rounds_df.at[idx, 'ct_losing_streak'] = ct_losing_streak
        rounds_df.at[idx, 't_losing_streak'] = t_losing_streak


    return rounds_df

def add_buy_type(row):

    if row['round_num'] in [1, 13]:
        return "Pistol", "Pistol"

    if row['CT_team_current_equip_value'] < 5000:
        ct_buy_type = "Full Eco"
    elif 5000 <= row['CT_team_current_equip_value'] < 10000:
        ct_buy_type = "Semi-Eco"
    elif 10000 <= row['CT_team_current_equip_value'] < 20000:
        ct_buy_type = "Semi-Buy"
    elif row['CT_team_current_equip_value'] >= 20000:
        ct_buy_type = "Full Buy"
    else:
        ct_buy_type = "Unknown"

    if row['T_team_current_equip_value'] < 5000:
        t_buy_type = "Full Eco"
    elif 5000 <= row['T_team_current_equip_value'] < 10000:
        t_buy_type = "Semi-Eco"
    elif 10000 <= row['T_team_current_equip_value'] < 20000:
        t_buy_type = "Semi-Buy"
    elif row['T_team_current_equip_value'] >= 20000:
        t_buy_type = "Full Buy"
    else:
        t_buy_type = "Unknown"

    return ct_buy_type, t_buy_type

def calculate_5v4_advantage(rounds_df, first_kills_df):

    # Makes sure the columns exists
    rounds_df['5v4_advantage'] = None

    # Checks what team got the first kill
    for idx, row in rounds_df.iterrows():
        round_num = row['round_num']

        # Filters the first kills DataFrame for the current round
        first_kill = first_kills_df[first_kills_df['round_num'] == round_num]

        if not first_kill.empty:
            # Gets the team that made the first kill
            killer_team = first_kill.iloc[0]['attacker_side']

            # Defines the advantage based on the killer team
            if killer_team == 'ct':
                rounds_df.at[idx, '5v4_advantage'] = 'ct'
            elif killer_team == 't':
                rounds_df.at[idx, '5v4_advantage'] = 't'

    return rounds_df

for file_name in os.listdir(folder_path):
    if file_name.endswith('.dem'):

        file_path = os.path.join(folder_path, file_name)
        dem = Demo(file_path)
        dem.parse(player_props=["team_clan_name","total_rounds_played", "current_equip_value", "ct_losing_streak", "t_losing_streak"])

        # Gets all the Players' steam_ids
        this_file_players_id = dem.events.get('player_spawn')
        this_file_players_id = this_file_players_id.with_columns(
            this_file_players_id['user_steamid'].cast(pl.Utf8)
        )
        this_file_players_id = this_file_players_id.to_pandas()
        this_file_players_id = this_file_players_id[['user_steamid', 'user_name']].drop_duplicates()
        players_id = pd.concat([players_id, this_file_players_id], ignore_index=True)
        players_id = players_id[['user_steamid', 'user_name']].drop_duplicates()

        # Grenades Data
        # Makes that the data frame is not empty and that the columns are in the right format
        this_file_flashes = dem.events.get('flashbang_detonate', pl.DataFrame())
        if this_file_flashes is not None and len(this_file_flashes) > 0:
            this_file_flashes = this_file_flashes.with_columns(
                this_file_flashes['user_steamid'].cast(pl.Utf8)
            )
        this_file_he = dem.events.get('hegrenade_detonate', pl.DataFrame())
        if this_file_he is not None and len(this_file_he) > 0:
            this_file_he = this_file_he.with_columns(
                this_file_he['user_steamid'].cast(pl.Utf8)
            )
        this_file_infernos = dem.events.get('inferno_startburn', pl.DataFrame())
        if this_file_infernos is not None and len(this_file_infernos) > 0:
            this_file_infernos = this_file_infernos.with_columns(
                this_file_infernos['user_steamid'].cast(pl.Utf8)
            )
        this_file_smoke = dem.events.get('smokegrenade_detonate', pl.DataFrame())
        if this_file_smoke is not None and len(this_file_smoke) > 0:
            this_file_smoke = this_file_smoke.with_columns(
                this_file_smoke['user_steamid'].cast(pl.Utf8)
            )
        this_file_util_dmg = dem.events.get('player_hurt', pl.DataFrame())
        if this_file_util_dmg is not None and len(this_file_util_dmg) > 0:
            this_file_util_dmg = this_file_util_dmg.with_columns(
                this_file_util_dmg['attacker_steamid'].cast(pl.Utf8)
            )
        util_dmg = this_file_util_dmg.filter(
            (this_file_util_dmg["weapon"] == "hegrenade") |
            (this_file_util_dmg["weapon"] == "molotov")   |
            (this_file_util_dmg["weapon"] == "inferno")
        )
        # Makes sure that the data frames are not empty, converts them to pandas and appends them to the main data frame
        if this_file_flashes is not None and len(this_file_flashes) > 0:
            df_flashes = pd.concat([df_flashes, this_file_flashes.to_pandas()], ignore_index=True)
        if this_file_he is not None and len(this_file_he) > 0:
            df_he = pd.concat([df_he, this_file_he.to_pandas()], ignore_index=True)
        if this_file_infernos is not None and len(this_file_infernos) > 0:
            df_infernos = pd.concat([df_infernos, this_file_infernos.to_pandas()], ignore_index=True)
        if this_file_smoke is not None and len(this_file_smoke) > 0:
            df_smoke = pd.concat([df_smoke, this_file_smoke.to_pandas()], ignore_index=True)
        if this_file_util_dmg is not None and len(this_file_util_dmg) > 0:
            df_util_dmg = pd.concat([df_util_dmg, this_file_util_dmg.to_pandas()], ignore_index=True)

        # Opening Kills Data
        this_file_df_kills = awpy.stats.calculate_trades(demo=dem)
        this_file_df_kills = this_file_df_kills.with_columns(
            this_file_df_kills['attacker_steamid'].cast(pl.Utf8),
            this_file_df_kills['assister_steamid'].cast(pl.Utf8),
            this_file_df_kills['victim_steamid'].cast(pl.Utf8)
        )
        this_file_df_kills = this_file_df_kills.to_pandas()
        first_kills = this_file_df_kills.sort_values(by=['round_num', 'tick'])
        first_kills = first_kills.groupby('round_num').first().reset_index()
        df_all_first_kills = pd.concat([df_all_first_kills, first_kills], ignore_index=True)

        # Rounds Data
        this_file_df_ticks = dem.ticks
        this_file_df_rounds = dem.rounds
        this_file_df_rounds = add_round_winners(this_file_df_ticks,this_file_df_rounds)
        this_file_df_rounds[['CT_buy_type', 'T_buy_type']] = this_file_df_rounds.apply(add_buy_type, axis=1, result_type='expand')
        first_kills = this_file_df_kills.sort_values(by=['round_num', 'tick'])
        first_kills = first_kills.groupby('round_num').first().reset_index()
        df_all_first_kills = pd.concat([df_all_first_kills, first_kills], ignore_index=True)   

        this_file_df_rounds = calculate_5v4_advantage(this_file_df_rounds, df_all_first_kills)
        this_file_df_rounds['demo_file_name'] = file_name
        df_rounds = pd.concat([df_rounds, this_file_df_rounds], ignore_index=True)

        # Creates Match Table
        df_matches = pd.concat([df_matches, pd.DataFrame({'file_id': [file_id_counter], 'file_name': [file_name]})], ignore_index=True)
        file_id_counter += 1

        # Creates rounds won columns
        this_file_team_rounds_won = this_file_df_rounds.groupby('winner_clan_name').agg(
            total_rounds_won=('winner_clan_name', 'size'),
            t_rounds_won=('winner', lambda x: (x == 'ct').sum()),
            ct_rounds_won=('winner', lambda x: (x == 't').sum())
        ).reset_index()
        this_file_team_rounds_won.columns = ['team_clan_name', 'total_rounds_won','t_rounds_won', 'ct_rounds_won']
        team_rounds_won = pd.concat([team_rounds_won,this_file_team_rounds_won], ignore_index=True)
        df_kills = pd.concat([df_kills,this_file_df_kills], ignore_index=True)

        # ADR Data
        this_file_adr = awpy.stats.adr(demo=dem)
        this_file_adr = this_file_adr.with_columns(
            this_file_adr['steamid'].cast(pl.Utf8)
        )
        this_file_adr = this_file_adr.to_pandas()
        this_file_adr = this_file_adr.drop(['adr', 'name'], axis=1)
        df_adr = pd.concat([df_adr, this_file_adr], ignore_index=True)
        df_adr = df_adr.groupby(['steamid','side'], as_index=False).sum()
        df_adr = df_adr[df_adr['side'] != 'all']


        # KAST Data
        this_file_kast = awpy.stats.kast(demo=dem)
        this_file_kast = this_file_kast.with_columns(
            this_file_kast['steamid'].cast(pl.Utf8)
        )
        this_file_kast = this_file_kast.to_pandas()
        this_file_kast = this_file_kast.drop(['kast', 'name'], axis=1)
        df_kast = pd.concat([df_kast, this_file_kast], ignore_index=True)
        df_kast = df_kast.groupby(['steamid','side'], as_index=False).sum()
        df_kast = df_kast[df_kast['side'] != 'all']

        print(f"{i}: Processed {file_name}")
        i = i + 1

player_kills = df_kills.groupby('attacker_steamid').agg(
    kills=('attacker_steamid', 'size'),
    headshots=('headshot', lambda x: (x == 1).sum()),
    wallbang_kills=('penetrated', lambda x: (x == 1).sum()),
    assisted_flashes=('assistedflash', lambda x: (x == 1).sum()),
    trade_kills=('was_traded', lambda x: (x == 1).sum()),
    no_scope=('noscope', lambda x: (x == 1).sum()),
    through_smoke=('thrusmoke', lambda x: (x == 1).sum()),
    airbone_kills=('attackerinair', lambda x: (x == 1).sum()),
    blind_kills=('attackerblind', lambda x: (x == 1).sum()),
    victim_blind_kills=('assistedflash', lambda x: (x == 1).sum()),
    attacker_team_clan_name=('attacker_team_clan_name', 'first'),
    awp_kills=('weapon', lambda x: (x == 'awp').sum()),
    pistol_kills=('weapon', lambda x: x.isin(['glock', 'usp_silencer', 'p250', 'p2000' , 'tec9', 'cz75_auto', 'fiveseven', 'deagle', 'elite', 'revolver']).sum())
).reset_index()
player_kills.rename(columns={'attacker_steamid': 'steam_id'}, inplace=True)
player_kills.rename(columns={'attacker_team_clan_name': 'team_clan_name'}, inplace=True)

player_assists = df_kills.groupby('assister_steamid').agg(
    assists=('assister_steamid', 'size')
).reset_index()
player_assists.rename(columns={'assister_steamid': 'steam_id'}, inplace=True)

player_deaths = df_kills.groupby('victim_steamid').agg(
    deaths=('victim_steamid', 'size'),
    trade_deaths=('was_traded', lambda x: (x == 1).sum())
).reset_index()
player_deaths.rename(columns={'victim_steamid': 'steam_id'}, inplace=True)

players = player_kills.merge(player_assists, on='steam_id', how='left').merge(player_deaths, on='steam_id', how='left')
players['steam_id'] = player_kills['steam_id'].astype('int64')
players['kd'] = players['kills'] / players['deaths']
players['k_d_diff'] = players['kills'] - players['deaths']


# ADR Total
adr_total = df_adr.groupby('steamid').agg({'dmg': 'sum', 'n_rounds': 'sum'})
adr_total['adr_total'] = adr_total['dmg'] / adr_total['n_rounds']
# ADR CT
df_ct = df_adr[df_adr['side'] == 'ct']
adr_ct = df_ct.groupby('steamid').agg({'dmg': 'sum', 'n_rounds': 'sum'})
adr_ct['adr_ct_side'] = adr_ct['dmg'] / adr_ct['n_rounds']
# ADR T
df_t = df_adr[df_adr['side'] == 't']
adr_t = df_t.groupby('steamid').agg({'dmg': 'sum', 'n_rounds': 'sum'})
adr_t['adr_t_side'] = adr_t['dmg'] / adr_t['n_rounds']
# ADR Data Frame
df_adr = pd.DataFrame({
    'steamid': adr_total.index,
    'adr_total': adr_total['adr_total'],
    'adr_ct_side': adr_ct['adr_ct_side'].reindex(adr_total.index, fill_value=0),
    'adr_t_side': adr_t['adr_t_side'].reindex(adr_total.index, fill_value=0),
}).reset_index(drop=True)
# Merge ADR
df_adr.rename(columns={'steamid': 'steam_id'}, inplace=True)
df_adr['steam_id'] = df_adr['steam_id'].astype('int64')
players = players.merge(df_adr, on='steam_id', how='left')


# KAST Total
kast_total = df_kast.groupby('steamid').agg({'kast_rounds': 'sum', 'n_rounds': 'sum'})
kast_total['kast_total'] = kast_total['kast_rounds'] / kast_total['n_rounds']
# KAST CT
df_ct = df_kast[df_kast['side'] == 'ct']
kast_ct = df_ct.groupby('steamid').agg({'kast_rounds': 'sum', 'n_rounds': 'sum'})
kast_ct['kast_ct_side'] = kast_ct['kast_rounds'] / kast_ct['n_rounds']
# KAST T
df_t = df_kast[df_kast['side'] == 't']
kast_t = df_t.groupby('steamid').agg({'kast_rounds': 'sum', 'n_rounds': 'sum'})
kast_t['kast_t_side'] = kast_t['kast_rounds'] / kast_t['n_rounds']
# KAST Data Frame
df_kast = pd.DataFrame({
    'steamid': kast_total.index,
    'kast_total': kast_total['kast_total'],
    'kast_ct_side': kast_ct['kast_ct_side'].reindex(kast_total.index, fill_value=0),
    'kast_t_side': kast_t['kast_t_side'].reindex(kast_total.index, fill_value=0),
}).reset_index(drop=True)
# Merge KAST
df_kast.rename(columns={'steamid': 'steam_id'}, inplace=True)
df_kast['steam_id'] = df_kast['steam_id'].astype('int64')
players = players.merge(df_kast, on='steam_id', how='left')

# Rounds won Data
team_rounds_won = team_rounds_won.groupby('team_clan_name').agg(
    total_rounds_won=('total_rounds_won', 'sum'),
    t_rounds_won=('t_rounds_won', 'sum'),
    ct_rounds_won=('ct_rounds_won', 'sum')
).reset_index()

players = players.merge(team_rounds_won, on='team_clan_name', how='left')

# Opening Duels
# Creates the player table with all the kills raw data grouped for every match in a single table
opening_kills = df_all_first_kills.groupby('attacker_steamid').agg(
    first_kills=('attacker_steamid', 'size'),
    ct_first_kills=('attacker_side', lambda x: (x == 'ct').sum()),
    t_first_kills=('attacker_side', lambda x: (x == 't').sum())
).reset_index()
opening_kills.rename(columns={'attacker_steamid': 'steam_id'}, inplace=True)
opening_kills['steam_id'] = opening_kills['steam_id'].astype('int64')

opening_deaths = df_all_first_kills.groupby('victim_steamid').agg(
    first_deaths=('victim_steamid', 'size'),  
    ct_first_deaths=('victim_side', lambda x: (x == 'ct').sum()),  
    t_first_deaths=('victim_side', lambda x: (x == 't').sum())  
).reset_index()
opening_deaths.rename(columns={'victim_steamid': 'steam_id'}, inplace=True)
opening_deaths['steam_id'] = opening_deaths['steam_id'].astype('int64')

player_first_kills = opening_kills.merge(opening_deaths, on='steam_id', how='left')
players = players.merge(player_first_kills, on='steam_id', how='left')

# Group all the grenades data in separated dataframes
df_flashes.rename(columns={'user_total_rounds_played': 'round'}, inplace=True)
df_flashes['round'] = df_flashes['round'] + 1
df_all_flashes = df_flashes.groupby('user_steamid').agg(
    flahes_thrown=('user_steamid', 'size'),
    ct_flahes_thrown=('user_side', lambda x: (x == 'ct').sum()),
    t_flahes_thrown=('user_side', lambda x: (x == 't').sum()),
    flahes_thrown_in_pistol_round=('round', lambda x: ((x == 1) | (x == 13)).sum()),
).reset_index()
df_all_flashes.rename(columns={'user_steamid': 'steam_id'}, inplace=True)
df_all_flashes['steam_id'] = df_all_flashes['steam_id'].astype('int64')

df_he.rename(columns={'user_total_rounds_played': 'round'}, inplace=True)
df_he['round'] = df_he['round'] + 1
df_all_he = df_he.groupby('user_steamid').agg(
    he_thrown=('user_steamid', 'size'),
    ct_he_thrown=('user_side', lambda x: (x == 'ct').sum()),
    t_he_thrown=('user_side', lambda x: (x == 't').sum()),
    he_thrown_in_pistol_round=('round', lambda x: ((x == 1) | (x == 13)).sum())
).reset_index()
df_all_he.rename(columns={'user_steamid': 'steam_id'}, inplace=True)
df_all_he['steam_id'] = df_all_he['steam_id'].astype('int64')

df_infernos.rename(columns={'user_total_rounds_played': 'round'}, inplace=True)
df_infernos['round'] = df_infernos['round'] + 1
df_all_infernos = df_infernos.groupby('user_steamid').agg(
    infernos_thrown=('user_steamid', 'size'),
    ct_infernos_thrown=('user_side', lambda x: (x == 'ct').sum()),
    t_infernos_thrown=('user_side', lambda x: (x == 't').sum()),
    infernos_thrown_in_pistol_round=('round', lambda x: ((x == 1) | (x == 13)).sum())
).reset_index()
df_all_infernos.rename(columns={'user_steamid': 'steam_id'}, inplace=True)
df_all_infernos['steam_id'] = df_all_infernos['steam_id'].astype('int64')

df_smoke.rename(columns={'user_total_rounds_played': 'round'}, inplace=True)
df_smoke['round'] = df_smoke['round'] + 1
df_all_smokes = df_smoke.groupby('user_steamid').agg(
    smokes_thrown=('user_steamid', 'size'),
    ct_smokes_thrown=('user_side', lambda x: (x == 'ct').sum()),
    t_smokes_thrown=('user_side', lambda x: (x == 't').sum()),
    smokes_thrown_in_pistol_round=('round', lambda x: ((x == 1) | (x == 13)).sum())
).reset_index()
df_all_smokes.rename(columns={'user_steamid': 'steam_id'}, inplace=True)
df_all_smokes['steam_id'] = df_all_smokes['steam_id'].astype('int64')

# Filter the utility damage data to only include the relevant columns and group by player
df_util_dmg = df_util_dmg[(df_util_dmg["weapon"] == "hegrenade") | (df_util_dmg["weapon"] == "molotov") | (df_util_dmg["weapon"] == "inferno")]
df_util_dmg.loc[:, "total_dmg"] = df_util_dmg["dmg_health"]
df_util_dmg = df_util_dmg.groupby("attacker_steamid").agg(
    total_util_dmg=("total_dmg", "sum"),
    ct_total_util_dmg=("total_dmg", lambda x: x[df_util_dmg.loc[x.index, "attacker_side"] == "ct"].sum()),
    t_total_util_dmg=("total_dmg", lambda x: x[df_util_dmg.loc[x.index, "attacker_side"] == "t"].sum())
).reset_index()
df_util_dmg.rename(columns={'attacker_steamid': 'steam_id'}, inplace=True)
df_util_dmg['steam_id'] = df_util_dmg['steam_id'].astype('int64')

# Merges all the current data in a single dataframe
players = players.merge(df_all_flashes, on='steam_id', how='left')
players = players.merge(df_all_he, on='steam_id', how='left')
players = players.merge(df_all_infernos, on='steam_id', how='left')
players = players.merge(df_all_smokes, on='steam_id', how='left')
players['util_in_pistol_round'] = players['flahes_thrown_in_pistol_round'] + players['he_thrown_in_pistol_round'] + players['infernos_thrown_in_pistol_round'] +  players['smokes_thrown_in_pistol_round']
players['total_util_thrown'] = players['flahes_thrown'] + players['he_thrown'] + players['infernos_thrown'] +  players['smokes_thrown']
players = players.merge(df_util_dmg, on='steam_id', how='left')

players_id.rename(columns={'user_steamid': 'steam_id'}, inplace=True)
players_id['steam_id'] = players_id['steam_id'].fillna(0).astype('int64')
players_id['steam_id'] = players_id['steam_id'].astype('int64')
players = players.merge(players_id, on='steam_id', how='left')
players = players[["steam_id", "user_name"] + [col for col in players.columns if col not in ["steam_id", "user_name"]]]
players.to_csv('Data_Export.csv')
df_rounds.to_csv('Rounds.csv')
df_matches.to_csv('Matches.csv')