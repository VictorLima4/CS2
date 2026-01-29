# Imports
import pandas as pd
import numpy as np
import polars as pl
import os
import time
import awpy
import re
from demoparser2 import DemoParser
from awpy import Demo
from pathlib import Path
from awpy.stats import adr
from awpy.stats import kast
from awpy.stats import rating
from awpy.stats import calculate_trades
from supabase import create_client, Client
from dotenv import load_dotenv
from requests import post, get
from datetime import datetime

# Custom Imports
from team_normalizer import normalize_team_name, normalization_log, get_normalization_stats, save_normalization_log, clear_log
from cs_demo_functions import (
    add_round_winners, add_losing_streaks, add_buy_type,
    calculate_advantage_5v4, insert_table, insert_or_update_player_history,
    rounds_correction, fetch_all_rows, calculate_clutches,
    calculate_multikill_rounds, extract_game_map, get_match_winner_team_name,
    get_file_modified_date, is_valid_dem, convert_to_nullable_int
)

start = time.time()

# Desktop Path
folder_path = r'C:\Users\bayli\Documents\CS Demos\debugging'

# # Notebook Path
# folder_path = r'G:\Meu Drive\Documents\CS Demos\test_demos'

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
df_clutches = pd.DataFrame()    
df_multikills = pd.DataFrame()
team_rounds_won = pd.DataFrame()
players_id = pd.DataFrame()
df_matches = pd.DataFrame(columns=['event_id','match_name'])
i = 1
current_schema = "staging"
event_id = 2

clear_log()

# Main Loop
for file_name in os.listdir(folder_path):
    if file_name.endswith('.dem'):

        file_path = os.path.join(folder_path, file_name)

        # Quick validation to avoid passing truncated/corrupt demos to awpy's parser which raises
        # OutOfBytesError when the demo is incomplete. If validation fails we skip and continue.
        if not is_valid_dem(file_path):
            print(f"[!] Skipping {file_name}: file seems too small or missing expected header (possible corruption)")
            continue

        # Wrap parser instantiation and parse in try/except so a single bad demo doesn't crash the whole run
        try:
            dem = Demo(file_path)
            dem.parse(player_props=["team_clan_name","total_rounds_played", "current_equip_value", "ct_losing_streak", "t_losing_streak", "is_alive"])
        except Exception as e:
            # Catch broad Exception here because the parser surfaces OutOfBytesError and related parser errors
            print(f"[!] Error parsing {file_name}: {e}")
            continue
        folder_name = os.path.basename(os.path.dirname(file_path))

        # Gets all the Players' steam_ids
        this_file_players_id = dem.events.get('player_spawn')
        this_file_players_id = this_file_players_id.with_columns(
            this_file_players_id['user_steamid'].cast(pl.Utf8)
        )
        this_file_players_id = this_file_players_id.to_pandas()
        this_file_players_id = this_file_players_id[['user_steamid', 'user_name']].drop_duplicates()
        if not this_file_players_id.empty:
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
        if not first_kills.empty:
            df_all_first_kills = pd.concat([df_all_first_kills, first_kills], ignore_index=True)

        # Rounds Data
        this_file_df_ticks = dem.ticks
        this_file_df_rounds = dem.rounds

        # Correction needed for the bomb_site column due to bug in awpy library
        this_file_bomb_planted = dem.events.get('bomb_planted', pl.DataFrame())

        # Ensure the rounds frame has a `bomb_plant` column and that it's nullable Int64
        if 'bomb_plant' not in this_file_df_rounds.columns:
            this_file_df_rounds = this_file_df_rounds.with_columns(
                pl.lit(None).alias('bomb_plant')
            )
        this_file_df_rounds = this_file_df_rounds.with_columns(
            pl.col('bomb_plant').cast(pl.Int64)
        )

        # If bomb plant events exist and contain the needed columns, cast tick and join.
        if "tick" in this_file_bomb_planted.columns and "user_place" in this_file_bomb_planted.columns:
            this_file_bomb_planted = this_file_bomb_planted.with_columns(
                pl.col("tick").cast(pl.Int64)
            )

            # Perform left join; rows with null `bomb_plant` will keep `bomb_site` as null
            this_file_df_rounds = this_file_df_rounds.join(
                this_file_bomb_planted.select(['tick', 'user_place']),
                left_on='bomb_plant',
                right_on='tick',
                how='left'
            ).with_columns(
                pl.col('user_place').alias('bomb_site')
            ).drop('user_place')
        else:
            # If no bomb plant events, just add a bomb_site column with None
            this_file_df_rounds = this_file_df_rounds.with_columns(
                pl.lit(None).alias('bomb_site')
            )
        # End of correction

        this_file_df_rounds = rounds_correction(this_file_df_rounds)
        this_file_df_rounds = add_round_winners(this_file_df_ticks,this_file_df_rounds)
        this_file_df_rounds = add_losing_streaks(this_file_df_rounds)
        this_file_df_rounds[['ct_buy_type', 't_buy_type']] = this_file_df_rounds.apply(add_buy_type, axis=1, result_type='expand')
        first_kills = this_file_df_kills.sort_values(by=['round_num', 'tick'])
        first_kills = first_kills.groupby('round_num').first().reset_index()
        if not first_kills.empty:
            df_all_first_kills = pd.concat([df_all_first_kills, first_kills], ignore_index=True)   

        this_file_df_rounds = calculate_advantage_5v4(this_file_df_rounds, df_all_first_kills)
        this_file_df_rounds['match_name'] = file_name.replace(f"{folder_name}_", "")
        if not this_file_df_rounds.empty:
            df_rounds = pd.concat([df_rounds, this_file_df_rounds], ignore_index=True)
        df_rounds['event_id'] = event_id

        # Creates rounds won columns
        this_file_team_rounds_won = this_file_df_rounds.groupby('winner_clan_name').agg(
            total_rounds_won=('winner_clan_name', 'size'),
            t_rounds_won=('winner', lambda x: (x == 'ct').sum()),
            ct_rounds_won=('winner', lambda x: (x == 't').sum())
        ).reset_index()
        this_file_team_rounds_won.columns = ['team_clan_name', 'total_rounds_won','t_rounds_won', 'ct_rounds_won']
        if not this_file_team_rounds_won.empty:
            team_rounds_won = pd.concat([team_rounds_won,this_file_team_rounds_won], ignore_index=True)
        if not this_file_df_kills.empty:
            df_kills = pd.concat([df_kills,this_file_df_kills], ignore_index=True)

        # Creates Match Table
        file_name = file_name.replace(f"{folder_name}_", "")
        winner_team, defeated_team, winner_score, defeated_score = get_match_winner_team_name(this_file_df_rounds)
        file_date = get_file_modified_date(file_path)
        new_match_row = pd.DataFrame({
            'match_name': [file_name],
            'event_id': [event_id],
            'winner_team_name': [winner_team],
            'defeated_team_name': [defeated_team],
            'winner_score': [winner_score],
            'defeated_score': [defeated_score],
            'match_date': [file_date]
        })

        # Only concat if not all NA or empty
        if not new_match_row.empty and not new_match_row.isna().all(axis=1).iloc[0]:
            df_matches = pd.concat([df_matches, new_match_row], ignore_index=True)

        # Getting 'map_name' and 'game_number' from the file name
        df_matches[['game_number','map_name']] = df_matches['match_name'].apply(lambda x: pd.Series(extract_game_map(x)))

        # ADR Data
        this_file_adr = awpy.stats.adr(demo=dem)
        this_file_adr = this_file_adr.with_columns(
            this_file_adr['steamid'].cast(pl.Utf8)
        )
        this_file_adr = this_file_adr.to_pandas()
        this_file_adr = this_file_adr.drop(['adr', 'name'], axis=1)
        if not this_file_adr.empty:
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
        if not this_file_kast.empty:
            df_kast = pd.concat([df_kast, this_file_kast], ignore_index=True)
        df_kast = df_kast.groupby(['steamid','side'], as_index=False).sum()
        df_kast = df_kast[df_kast['side'] != 'all']

        # Creates Clutches Dataframe
        this_file_clutches = calculate_clutches(dem)
        # Guard against empty DataFrames or missing columns before selecting/casting
        # Use `pl.col([...])` to cast multiple columns safely
        required_cols = {"clutcher_steamid", "opponent_steamid"}
        if (hasattr(this_file_clutches, 'height') and this_file_clutches.height > 0) or required_cols.issubset(set(this_file_clutches.columns)):
            this_file_clutches = this_file_clutches.with_columns(
                pl.col(["clutcher_steamid", "opponent_steamid"]).cast(pl.Utf8)
            )
        this_file_clutches = this_file_clutches.to_pandas()
        this_file_clutches['match_name'] = file_name
        if not this_file_clutches.empty:
            df_clutches = pd.concat([df_clutches,this_file_clutches], ignore_index=True)
        # Only drop columns that actually exist to avoid KeyError when df_clutches is empty
        cols_to_drop = [c for c in ['clutcher_name', 'clutch_start_tick', 'clutch_end_tick'] if c in df_clutches.columns]
        if cols_to_drop:
            df_clutches = df_clutches.drop(cols_to_drop, axis=1)
        df_clutches['file_id'] = event_id
        df_clutches['event_id'] = event_id

        # Creates Multikills Dataframe
        this_file_multikills = calculate_multikill_rounds(dem)
        this_file_multikills = this_file_multikills.with_columns(
            this_file_multikills['steam_id'].cast(pl.Utf8)
        )
        this_file_multikills = this_file_multikills.to_pandas()
        if not this_file_multikills.empty:
            df_multikills = pd.concat([df_multikills, this_file_multikills], ignore_index=True)

        print(f"{i}: Processed {file_name}")
        i = i + 1

log_file = f'logs/team_normalizations_{folder_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
save_normalization_log(log_file)

print(f"\n✓ Processing complete. Normalization log saved to: {log_file}")

# Data cleaning and modeling

player_kills = df_kills.groupby('attacker_steamid').agg(
    kills=('attacker_steamid', 'size'),
    ct_kills=('attacker_side', lambda x: (x == 'ct').sum()),
    t_kills=('attacker_side', lambda x: (x == 't').sum()),
    headshots=('headshot', lambda x: (x == 1).sum()),
    wallbang_kills=('penetrated', lambda x: (x == 1).sum()),
    assisted_flashes=('assistedflash', lambda x: (x == 1).sum()),
    ct_assisted_flashes=('assistedflash', lambda x: ((x == 1) & (df_kills.loc[x.index, 'attacker_side'] == 'ct')).sum()),
    t_assisted_flashes=('assistedflash', lambda x: ((x == 1) & (df_kills.loc[x.index, 'attacker_side'] == 't')).sum()),
    trade_kills=('was_traded', lambda x: (x == 1).sum()),
    no_scope=('noscope', lambda x: (x == 1).sum()),
    through_smoke=('thrusmoke', lambda x: (x == 1).sum()),
    airborne_kills=('attackerinair', lambda x: (x == 1).sum()),
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

# Normalize team_clan_name in players dataframe to match database canonical names
players['team_clan_name'] = players['team_clan_name'].apply(lambda x: normalize_team_name(x)[0])

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
    flashes_thrown=('user_steamid', 'size'),
    ct_flashes_thrown=('user_side', lambda x: (x == 'ct').sum()),
    t_flashes_thrown=('user_side', lambda x: (x == 't').sum()),
    flashes_thrown_in_pistol_round=('round', lambda x: ((x == 1) | (x == 13)).sum()),
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

# Group the multikills data
df_multikills = df_multikills.groupby('steam_id')[['2k', '3k', '4k', '5k']].sum().reset_index()
df_multikills['steam_id'] = df_multikills['steam_id'].astype('int64')

# Merges all the current data in a single dataframe
players = players.merge(df_all_flashes, on='steam_id', how='left')
players = players.merge(df_all_he, on='steam_id', how='left')
players = players.merge(df_all_infernos, on='steam_id', how='left')
players = players.merge(df_all_smokes, on='steam_id', how='left')
players = players.merge(df_multikills, on='steam_id', how='left')
players['util_in_pistol_round'] = players['flashes_thrown_in_pistol_round'] + players['he_thrown_in_pistol_round'] + players['infernos_thrown_in_pistol_round'] +  players['smokes_thrown_in_pistol_round']
players['total_util_thrown'] = players['flashes_thrown'] + players['he_thrown'] + players['infernos_thrown'] +  players['smokes_thrown']
players = players.merge(df_util_dmg, on='steam_id', how='left')

players_id.rename(columns={'user_steamid': 'steam_id'}, inplace=True)
players_id['steam_id'] = players_id['steam_id'].fillna(0).astype('int64')
players_id['steam_id'] = players_id['steam_id'].astype('int64')
players = players.merge(players_id, on='steam_id', how='left')
players = players[["steam_id", "user_name"] + [col for col in players.columns if col not in ["steam_id", "user_name"]]]

players = convert_to_nullable_int(players, columns=[
    'kills', 'deaths', 'assists', 'headshots', 'wallbang_kills', 
    'trade_kills', 'trade_deaths', 'first_kills', 'ct_first_kills', 
    't_first_kills', 'first_deaths', 'ct_first_deaths', 't_first_deaths',
    'flashes_thrown', 'he_thrown', 'infernos_thrown', 'smokes_thrown',
    '2k', '3k', '4k', '5k', 'total_rounds_won', 't_rounds_won', 'ct_rounds_won'
])

# Convert float columns to nullable Float64
float_cols = ['adr_total', 'adr_ct_side', 'adr_t_side', 'kast_total', 'kast_ct_side', 'kast_t_side', 'kd', 'k_d_diff']
for col in float_cols:
    if col in players.columns:
        players[col] = players[col].astype('Float64')

# Creates the teams table
teams = pd.DataFrame({'team_clan_name': pd.concat([df_rounds['ct_team_clan_name'], df_rounds['t_team_clan_name']]).dropna().unique()})

# Database Management and Insertion
# Load environment variables from .env file
load_dotenv()
url = os.getenv("url")
key = os.getenv("key")
supabase: Client = create_client(url, key)

# DB Management
players_df = players[['steam_id', 'user_name', 'team_clan_name']].drop_duplicates().reset_index(drop=True)

df_rounds['bomb_plant'] = df_rounds['bomb_plant'].replace({np.nan: None})
df_rounds['bomb_site'] = df_rounds['bomb_site'].replace({np.nan: None})

# Insert teams_data into the database
teams_data = [{"team_clan_name": name} for name in teams]
insert_table(teams, supabase, current_schema, "teams", conflict_cols=["team_clan_name"])
# Gets the team_id created by supabase
teams = fetch_all_rows(supabase, current_schema, "teams")
team_id_map = {item['team_clan_name']: item['id'] for item in teams}
# Normalize team names in players_df before mapping to team_id
players_df["team_clan_name"] = players_df["team_clan_name"].apply(lambda x: normalize_team_name(x)[0])
# Map team_clan_name to team_id in players_df
players_df["team_id"] = players_df["team_clan_name"].map(team_id_map)
players_df["team_id"] = players_df["team_id"].astype('Int64')
players_df = players_df[["steam_id", "user_name", "team_id"]].drop_duplicates()
# Insert players_data into the database
players_table = players_df.drop(columns=["team_id"])
insert_table(players_table, supabase, current_schema, "players", conflict_cols=["steam_id"])
# Insert players_history into the database
players_history = players_df.drop(columns=["user_name"])
insert_table(players_history[["steam_id", "team_id"]], supabase, current_schema, "player_history", conflict_cols=["steam_id, team_id"])

# Insert matches_data into the database
df_matches["winner_team_id"] = df_matches["winner_team_name"].map(team_id_map).fillna(pd.NA).astype('Int64')
df_matches["defeated_team_id"] = df_matches["defeated_team_name"].map(team_id_map).fillna(pd.NA).astype('Int64')
df_matches['winner_score'] = df_matches['winner_score'].fillna(pd.NA).astype('Int64')
df_matches['defeated_score'] = df_matches['defeated_score'].fillna(pd.NA).astype('Int64')
df_matches = df_matches.drop(['winner_team_name', 'defeated_team_name'], axis=1)
insert_table(df_matches, supabase, current_schema, "matches", conflict_cols=["match_name, event_id"])
# Gets the file_id created by supabase
matches = fetch_all_rows(supabase, current_schema, "matches")
file_id_map = {item['match_name']: item['file_id'] for item in matches}
# Map match_name to file_id in df_rounds and df_clutches
df_rounds["file_id"] = df_rounds["match_name"].map(file_id_map)
df_clutches["file_id"] = df_clutches["match_name"].map(file_id_map)
df_clutches = df_clutches.replace({np.nan: None})
# Insert rounds into the database
df_rounds = df_rounds.replace({np.nan: None})
insert_table(df_rounds, supabase, current_schema, "rounds", conflict_cols=["round_num, file_id"])
# Insert clutches_data into the database
insert_table(df_clutches, supabase, current_schema, "clutches_data", conflict_cols=["round_num, file_id"])

# Reads the teams table from the database
teams = pd.DataFrame(fetch_all_rows(supabase, current_schema, "teams"))
df_rounds = pd.DataFrame(fetch_all_rows(supabase, current_schema, "rounds"))

# Creates the player_match_summary table
player_match_summary = df_rounds[['file_id', 'ct_team_clan_name', 't_team_clan_name']].drop_duplicates()
players["steam_id"] = players["steam_id"].astype("Int64")

# Adds team ids for CT and T teams
player_match_summary = player_match_summary.merge(
    teams,
    left_on='ct_team_clan_name',
    right_on='team_clan_name',
    how='left'
).rename(columns={'id': 'CT_team_id'}).drop(columns=['team_clan_name'])

player_match_summary = player_match_summary.merge(
    teams,
    left_on='t_team_clan_name',
    right_on='team_clan_name',
    how='left'
).rename(columns={'id': 'T_team_id'}).drop(columns=['team_clan_name'])

# Adds the players' steam ids to the player_match_summary table
player_match_summary = player_match_summary.merge(
    players[['steam_id', 'team_clan_name']],
    left_on='ct_team_clan_name',
    right_on='team_clan_name',
    how='left'
).rename(columns={'steam_id': 'CT_steam_id'})

player_match_summary = player_match_summary.merge(
    players[['steam_id', 'team_clan_name']],
    left_on='t_team_clan_name',
    right_on='team_clan_name',
    how='left'
).rename(columns={'steam_id': 'T_steam_id'})

# Combines the CT and T players into a single table
player_match_summary = pd.concat([
    player_match_summary[['file_id', 'CT_steam_id', 'CT_team_id']].rename(
        columns={'CT_steam_id': 'steam_id', 'CT_team_id': 'team_id'}
    ),
    player_match_summary[['file_id', 'T_steam_id', 'T_team_id']].rename(
        columns={'T_steam_id': 'steam_id', 'T_team_id': 'team_id'}
    )
], ignore_index=True)

# Drops duplicates values
player_match_summary.loc[:, 'event_id'] = event_id
player_match_summary = player_match_summary.dropna().drop_duplicates()

# Creates 3 separated dataframes for kills, general stats and utility stats
kill_stats_cols = [
    'steam_id', 'kills', 'headshots', 'wallbang_kills', 'no_scope',
    'through_smoke', 'airborne_kills', 'blind_kills', 'victim_blind_kills',
    'awp_kills', 'pistol_kills', 'first_kills', 'ct_first_kills', 't_first_kills',
    'first_deaths', 'ct_first_deaths', 't_first_deaths', 'ct_kills', 't_kills', 'ct_assisted_flashes',
    't_assisted_flashes', '2k', '3k', '4k', '5k'
]
kill_stats_df = players[kill_stats_cols].copy()
kill_stats_df.loc[:, 'event_id'] = event_id
# Convert to Int64
kill_int_cols = ['kills', 'headshots', 'wallbang_kills', 'no_scope', 'through_smoke', 'airborne_kills', 
                 'blind_kills', 'victim_blind_kills', 'awp_kills', 'pistol_kills', 'first_kills', 'ct_first_kills', 
                 't_first_kills', 'first_deaths', 'ct_first_deaths', 't_first_deaths', 'ct_kills', 't_kills', 
                 'ct_assisted_flashes', 't_assisted_flashes', '2k', '3k', '4k', '5k']
kill_stats_df[kill_int_cols] = kill_stats_df[kill_int_cols].fillna(pd.NA).astype('Int64')

general_stats_cols = [
    'steam_id', 'assists', 'deaths', 'trade_kills', 'trade_deaths', 'kd', 'k_d_diff',
    'adr_total', 'adr_ct_side', 'adr_t_side', 'kast_total', 'kast_ct_side',
    'kast_t_side', 'total_rounds_won', 't_rounds_won', 'ct_rounds_won'
]
general_stats_df = players[general_stats_cols].copy()
general_stats_df.loc[:, 'event_id'] = event_id
# Convert Int64 and Float64 columns
general_int_cols = ['assists', 'deaths', 'trade_kills', 'trade_deaths', 'total_rounds_won', 't_rounds_won', 'ct_rounds_won']
general_stats_df[general_int_cols] = general_stats_df[general_int_cols].fillna(pd.NA).astype('Int64')
general_float_cols = ['kd', 'k_d_diff', 'adr_total', 'adr_ct_side', 'adr_t_side', 'kast_total', 'kast_ct_side', 'kast_t_side']
for col in general_float_cols:
    general_stats_df[col] = general_stats_df[col].replace([np.inf, -np.inf], np.nan).astype('Float64')

utility_stats_cols = [
    'steam_id', 'assisted_flashes', 'flashes_thrown', 'ct_flashes_thrown', 't_flashes_thrown',
    'flashes_thrown_in_pistol_round', 'he_thrown', 'ct_he_thrown', 't_he_thrown',
    'he_thrown_in_pistol_round', 'infernos_thrown', 'ct_infernos_thrown', 't_infernos_thrown',
    'infernos_thrown_in_pistol_round', 'smokes_thrown', 'ct_smokes_thrown', 't_smokes_thrown',
    'smokes_thrown_in_pistol_round', 'util_in_pistol_round', 'total_util_thrown', 'total_util_dmg', 'ct_total_util_dmg', 't_total_util_dmg'
]
utility_stats_df = players[utility_stats_cols].copy()
utility_stats_df.loc[:, 'event_id'] = event_id
# Convert Int64 and Float64 columns
utility_stats_cols = [
    'steam_id', 'assisted_flashes', 'flashes_thrown', 'ct_flashes_thrown', 't_flashes_thrown',
    'flashes_thrown_in_pistol_round', 'he_thrown', 'ct_he_thrown', 't_he_thrown',
    'he_thrown_in_pistol_round', 'infernos_thrown', 'ct_infernos_thrown', 't_infernos_thrown',
    'infernos_thrown_in_pistol_round', 'smokes_thrown', 'ct_smokes_thrown', 't_smokes_thrown',
    'smokes_thrown_in_pistol_round', 'util_in_pistol_round', 'total_util_thrown', 'total_util_dmg', 'ct_total_util_dmg', 't_total_util_dmg'
]
utility_stats_df = players[utility_stats_cols].copy()
utility_stats_df.loc[:, 'event_id'] = event_id
# Convert Int64 and Float64 columns
utility_int_cols = ['assisted_flashes', 'flashes_thrown', 'ct_flashes_thrown', 't_flashes_thrown', 'flashes_thrown_in_pistol_round',
                    'he_thrown', 'ct_he_thrown', 't_he_thrown', 'he_thrown_in_pistol_round', 'infernos_thrown', 'ct_infernos_thrown',
                    't_infernos_thrown', 'infernos_thrown_in_pistol_round', 'smokes_thrown', 'ct_smokes_thrown', 't_smokes_thrown',
                    'smokes_thrown_in_pistol_round', 'util_in_pistol_round', 'total_util_thrown', 'total_util_dmg', 'ct_total_util_dmg', 't_total_util_dmg']
utility_stats_df[utility_int_cols] = utility_stats_df[utility_int_cols].fillna(pd.NA).astype('Int64')

insert_table(kill_stats_df, supabase, current_schema, "kill_stats", conflict_cols=["steam_id, event_id"])
insert_table(general_stats_df, supabase, current_schema, "general_stats", conflict_cols=["steam_id, event_id"])
insert_table(utility_stats_df, supabase, current_schema, "utility_stats", conflict_cols=["steam_id, event_id"])
insert_table(player_match_summary, supabase, current_schema, "player_match_summary", conflict_cols=["file_id, steam_id, event_id"])

# # Data export
# players.to_csv(f'C:\\Users\\bayli\\Documents\\Git Projects\\CS2\\CSV\\data_export_{folder_name}.csv')
# df_rounds.to_csv(f'C:\\Users\\bayli\\Documents\\Git Projects\\CS2\\CSV\\rounds_{folder_name}.csv')
# df_clutches.to_csv(f'C:\\Users\\bayli\\Documents\\Git Projects\\CS2\\CSV\\clutches_{folder_name}.csv')
# df_matches.to_csv(f'matches_{folder_name}.csv')
# teams.to_csv(f'teams_{folder_name}.csv')
# player_match_summary.to_csv(f'player_match_summary_{folder_name}.csv')

end = time.time()
elapsed = int(end - start)
hours, remainder = divmod(elapsed, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Processed {i} files in {hours:02d}h{minutes:02d}m{seconds:02d}s")