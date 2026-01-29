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

# Functions

def add_round_winners(ticks_df, rounds_df):
    ticks_df = ticks_df.to_pandas()
    rounds_df = rounds_df.to_pandas()

    # Makes sure the columns exists
    rounds_df['ct_team_clan_name'] = None
    rounds_df['t_team_clan_name'] = None
    rounds_df['winner_clan_name'] = None
    rounds_df['ct_team_current_equip_value'] = None
    rounds_df['t_team_current_equip_value'] = None
    rounds_df['ct_losing_streak'] = None
    rounds_df['t_losing_streak'] = None

    for idx, row in rounds_df.iterrows():
        freeze_end_tick = row['freeze_end']
        winner = row['winner']

        # Takes all corresponding entries
        first_tick_df = ticks_df[ticks_df['tick'] == freeze_end_tick]

        # Takes the name for every team
        try:
            CT_team_raw = first_tick_df[first_tick_df['side'] == 'ct']['team_clan_name'].iloc[0]
            CT_team, _ = normalize_team_name(CT_team_raw)
        except IndexError:
            CT_team = None
        
        try:
            T_team_raw = first_tick_df[first_tick_df['side'] == 't']['team_clan_name'].iloc[0]
            T_team, _ = normalize_team_name(T_team_raw)
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

        # Determines the winner team name
        if winner == 'ct':
            winner_clan = CT_team
        elif winner in ['t', 'TERRORIST']:
            winner_clan = T_team
        else:
            winner_clan = None
            print(f"[!] Round {idx} - winner error: '{winner}'")
            
        # Fill Columns in the DataFrame
        rounds_df.at[idx, 'ct_team_clan_name'] = CT_team
        rounds_df.at[idx, 't_team_clan_name'] = T_team
        rounds_df.at[idx, 'winner_clan_name'] = winner_clan
        rounds_df.at[idx, 'ct_team_current_equip_value'] = CT_team_current_equip_value
        rounds_df.at[idx, 't_team_current_equip_value'] = T_team_current_equip_value


    return rounds_df

def add_losing_streaks(df: pd.DataFrame) -> pd.DataFrame:
    ct_losing_streak = []
    t_losing_streak = []

    ct_streak = 0
    t_streak = 0

    for _, row in df.iterrows():
        ct_team = row['ct_team_clan_name']
        t_team = row['t_team_clan_name']
        winner = row['winner_clan_name']
        
        if winner == ct_team:
            ct_streak = 0
            t_streak += 1
        else:  # winner == t_team
            t_streak = 0
            ct_streak += 1

        ct_losing_streak.append(ct_streak)
        t_losing_streak.append(t_streak)

    df['ct_losing_streak'] = ct_losing_streak
    df['t_losing_streak'] = t_losing_streak

    return df

def add_buy_type(row):

    if row['round_num'] in [1, 13]:
        return "Pistol", "Pistol"

    if row['ct_team_current_equip_value'] < 5000:
        ct_buy_type = "Full Eco"
    elif 5000 <= row['ct_team_current_equip_value'] < 10000:
        ct_buy_type = "Semi-Eco"
    elif 10000 <= row['ct_team_current_equip_value'] < 20000:
        ct_buy_type = "Semi-Buy"
    elif row['ct_team_current_equip_value'] >= 20000:
        ct_buy_type = "Full Buy"
    else:
        ct_buy_type = "Unknown"

    if row['t_team_current_equip_value'] < 5000:
        t_buy_type = "Full Eco"
    elif 5000 <= row['t_team_current_equip_value'] < 10000:
        t_buy_type = "Semi-Eco"
    elif 10000 <= row['t_team_current_equip_value'] < 20000:
        t_buy_type = "Semi-Buy"
    elif row['t_team_current_equip_value'] >= 20000:
        t_buy_type = "Full Buy"
    else:
        t_buy_type = "Unknown"

    return ct_buy_type, t_buy_type

def calculate_advantage_5v4(rounds_df, first_kills_df):

    # Makes sure the columns exists
    rounds_df['advantage_5v4'] = None

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
                rounds_df.at[idx, 'advantage_5v4'] = 'ct'
            elif killer_team == 't':
                rounds_df.at[idx, 'advantage_5v4'] = 't'

    return rounds_df

def insert_table(df, supabase, current_schema, table_name, conflict_cols):
    for idx, row in enumerate(df.to_dict(orient="records")):
        try:
            supabase.schema(current_schema).table(table_name).upsert(row, on_conflict=conflict_cols).execute()
        except Exception as e:
            print(f"\n[ERROR] Failed to insert row {idx} into {table_name}")
            print(f"[ERROR] Exception: {e}")
            print(f"[ERROR] Row data:")
            for col, val in row.items():
                print(f"  {col}: {val!r} ({type(val).__name__})")
            raise

def insert_or_update_player_history(players_df, supabase):
    for _, row in players_df.iterrows():
        steam_id = row["steam_id"]
        team_id = row["team_id"]
        
        player_history_data = {
            "steam_id": steam_id,
            "team_id": team_id
        }
        supabase.table("player_history").upsert(
            player_history_data, 
            on_conflict=["steam_id", "team_id"]
        ).execute()

def rounds_correction(df: pl.DataFrame) -> pl.DataFrame:
    freeze_end_is_null = df.select(pl.col("freeze_end").first().is_null()).item()

    if freeze_end_is_null:
        df = df.with_columns(
            (pl.col("round_num") - 1).alias("round_num")
        )
        
    return df.filter(pl.col("freeze_end").is_not_null())

def fetch_all_rows(supabase, current_schema, table_name, page_size=1000):
    offset = 0
    all_data = []

    while True:
        response = supabase.schema(current_schema).table(table_name).select("*").range(offset, offset + page_size - 1).execute()
        data = response.data
        if not data:
            break
        all_data.extend(data)
        offset += page_size

    return all_data

def calculate_clutches(dem) -> pl.DataFrame:
    clutches_data = []

    # Apply rounds_correction to dem.rounds
    corrected_rounds = rounds_correction(dem.rounds)

    # Map player SteamIDs to names
    all_players = dem.ticks.select(["steamid", "name"]).unique()
    player_name_map = {row["steamid"]: row["name"] for row in all_players.iter_rows(named=True)}

    for round_info in corrected_rounds.iter_rows(named=True):
        round_num = round_info['round_num']
        round_start_tick = round_info['start']
        round_end_tick = round_info['end']
        winning_team = round_info['winner']

        # Filter round ticks and kills
        round_ticks_df = dem.ticks.filter(
            (pl.col("tick") >= round_start_tick) & (pl.col("tick") <= round_end_tick)
        )
        round_kills_df = dem.kills.filter(
            (pl.col("round_num") == round_num) &
            (pl.col("tick") >= round_start_tick) & (pl.col("tick") <= round_end_tick)
        ).sort("tick")

        # Alive counts per tick
        alive_counts_per_tick = round_ticks_df.group_by("tick").agg(
            (pl.when(pl.col("side") == "ct").then(pl.col("is_alive")).otherwise(0)).sum().alias("ct_alive"),
            (pl.when(pl.col("side") == "t").then(pl.col("is_alive")).otherwise(0)).sum().alias("t_alive")
        )

        # Player states per tick
        players_alive_at_tick = round_ticks_df.group_by("tick").agg(
            pl.struct(["steamid", "is_alive", "side"]).alias("players_state")
        )

        merged_round_data = alive_counts_per_tick.join(
            players_alive_at_tick, on="tick", how="left"
        ).sort("tick")

        # Variables tracking clutch
        clutch_active = False
        clutcher_steamid = None
        clutcher_team_side = None
        clutch_start_tick = None
        opponents_at_start = 0
        one_v_one_reached = False
        last_opponent_steamid = None

        for tick_data in merged_round_data.iter_rows(named=True):
            current_tick = tick_data['tick']
            ct_alive = tick_data['ct_alive']
            t_alive = tick_data['t_alive']
            players_state = tick_data['players_state']

            is_clutch_condition_met = (ct_alive == 1 and t_alive >= 2) or \
                                      (t_alive == 1 and ct_alive >= 2)

            # Check if clutcher is still alive
            clutcher_is_alive_this_tick = False
            if clutch_active and clutcher_steamid:
                for p_state in players_state:
                    if p_state['steamid'] == clutcher_steamid and p_state['is_alive']:
                        clutcher_is_alive_this_tick = True
                        break

            # --- Start of a clutch ---
            if is_clutch_condition_met:
                if not clutch_active:
                    clutch_active = True
                    clutch_start_tick = current_tick
                    one_v_one_reached = False
                    last_opponent_steamid = None

                    if ct_alive == 1:
                        clutcher_team_side = 'ct'
                        opponents_at_start = t_alive
                    else:
                        clutcher_team_side = 't'
                        opponents_at_start = ct_alive

                    for p_state in players_state:
                        if p_state['is_alive'] and p_state['side'] == clutcher_team_side:
                            clutcher_steamid = p_state['steamid']
                            break

            else:
                # --- End of a clutch due to clutcher's death ---
                if clutch_active:
                    if not clutcher_is_alive_this_tick:
                        clutch_end_tick = current_tick
                        clutch_outcome = "lost"
                        final_clutch_kills = 0

                        if clutcher_steamid:
                            final_clutch_kills = round_kills_df.filter(
                                (pl.col("tick") >= clutch_start_tick) &
                                (pl.col("tick") <= clutch_end_tick) &
                                (pl.col("attacker_steamid") == clutcher_steamid)
                            ).shape[0]

                        round_team_won = (clutcher_team_side == winning_team.lower())

                        clutches_data.append({
                            "round_num": round_num,
                            "clutcher_steamid": clutcher_steamid,
                            "clutcher_name": player_name_map.get(clutcher_steamid, "Unknown"),
                            "clutcher_team_side": clutcher_team_side,
                            "opponents_at_start": opponents_at_start,
                            "clutch_start_tick": clutch_start_tick,
                            "clutch_end_tick": clutch_end_tick,
                            "clutch_kills": final_clutch_kills,
                            "clutch_outcome": clutch_outcome,
                            "round_won": round_team_won,
                            "clutcher_survived_round": False,
                            "1v1_situation": one_v_one_reached,
                            "opponent_steamid": last_opponent_steamid
                        })

                        # Reset
                        clutch_active = False
                        clutcher_steamid = None
                        clutcher_team_side = None
                        clutch_start_tick = None
                        opponents_at_start = 0
                        one_v_one_reached = False
                        last_opponent_steamid = None

            # --- NEW: Detect 1v1 situation while clutch is active ---
            if clutch_active and not one_v_one_reached:
                if ct_alive == 1 and t_alive == 1:
                    one_v_one_reached = True
                    for p_state in players_state:
                        if p_state['side'] != clutcher_team_side and p_state['is_alive']:
                            last_opponent_steamid = p_state['steamid']
                            break

        # --- Handle active clutch at end of round ---
        if clutch_active:
            clutch_end_tick = round_end_tick
            final_clutch_kills = 0

            if clutcher_steamid:
                final_clutch_kills = round_kills_df.filter(
                    (pl.col("tick") >= clutch_start_tick) &
                    (pl.col("tick") <= clutch_end_tick) &
                    (pl.col("attacker_steamid") == clutcher_steamid)
                ).shape[0]

            round_team_won = (clutcher_team_side == winning_team.lower())

            clutch_outcome = "won" if round_team_won else "save"
            clutcher_survived_round = True

            clutches_data.append({
                "round_num": round_num,
                "clutcher_steamid": clutcher_steamid,
                "clutcher_name": player_name_map.get(clutcher_steamid, "Unknown"),
                "clutcher_team_side": clutcher_team_side,
                "opponents_at_start": opponents_at_start,
                "clutch_start_tick": clutch_start_tick,
                "clutch_end_tick": clutch_end_tick,
                "clutch_kills": final_clutch_kills,
                "clutch_outcome": clutch_outcome,
                "round_won": round_team_won,
                "clutcher_survived_round": clutcher_survived_round,
                "1v1_situation": one_v_one_reached,
                "opponent_steamid": last_opponent_steamid
            })

            # Reset after round end
            clutch_active = False
            clutcher_steamid = None
            clutcher_team_side = None
            clutch_start_tick = None
            opponents_at_start = 0
            one_v_one_reached = False
            last_opponent_steamid = None

    clutches_df = pl.DataFrame(clutches_data)
    return clutches_df

def calculate_multikill_rounds(dem) -> pl.DataFrame:
    # Ensure 'dem.kills' is not empty before proceeding
    if dem.kills is None or dem.kills.shape[0] == 0:
        return pl.DataFrame({"steam_id": pl.Series(dtype=pl.UInt64),
                             "2k": pl.Series(dtype=pl.UInt32),
                             "3k": pl.Series(dtype=pl.UInt32),
                             "4k": pl.Series(dtype=pl.UInt32),
                             "5k": pl.Series(dtype=pl.UInt32)})

    kills_per_player_per_round = dem.kills.group_by(["round_num", "attacker_steamid"]).agg(
        pl.len().alias("kills_in_round")
    )

    multikill_counts = kills_per_player_per_round.group_by("attacker_steamid").agg(
        pl.when(pl.col("kills_in_round") == 2).then(1).otherwise(0).sum().alias("2k"),
        pl.when(pl.col("kills_in_round") == 3).then(1).otherwise(0).sum().alias("3k"),
        pl.when(pl.col("kills_in_round") == 4).then(1).otherwise(0).sum().alias("4k"),
        pl.when(pl.col("kills_in_round") == 5).then(1).otherwise(0).sum().alias("5k")
    )

    multikill_counts = multikill_counts.rename({"attacker_steamid": "steam_id"})

    return multikill_counts

def extract_game_map(fname):
    # Remove extension
    fname = re.sub(r'\.dem$', '', fname, flags=re.IGNORECASE)

    # Strip Windows duplicate suffixes like " (1)" that get appended to filenames
    fname = re.sub(r"\s*\(\d+\)$", "", fname)

    # Split tokens and clean trailing duplicate suffixes from each token
    parts = [re.sub(r"\s*\(\d+\)$", "", p).strip() for p in fname.split('-')]

    # Common CS map names (lowercase). Extend if you use custom maps
    MAP_NAMES = {
        'ancient', 'anubis', 'overpass', 'mirage', 'dust2', 'nuke', 'train',
        'vertigo', 'inferno', 'cache', 'cobblestone', 'cobble', 'season', 'de_cbble'
    }

    # 1) If a match token like m1..m5 exists, prefer it and the following token as the map
    for i, token in enumerate(parts):
        # Match only plausible game tokens m1..m5 (avoid matching team names like m80)
        if re.fullmatch(r'(?i)m(?:[1-9]|1[0-3])', token):
            game = token
            # Candidate for map is the next token
            map_part = parts[i+1] if i+1 < len(parts) else None
            if map_part:
                mp_lower = map_part.lower()
                # Accept if it's a known map or looks like a normal map token
                if mp_lower in MAP_NAMES or (re.fullmatch(r'[a-z0-9_]+', mp_lower) and len(mp_lower) >= 3 and mp_lower not in ('vs', 'v')):
                    return game, map_part
            # Found a game token but couldn't confidently pick a map -> keep searching

    # 2) No explicit game token found — try to find a map token anywhere (prefer the last token)
    # Many filenames are like "teamA-vs-teamB-mapname" where the map is the last token.
    # Scan tokens from the right and pick the first plausible map-looking token that is not a vs separator.
    for token in reversed(parts):
        if not token:
            continue
        t = token.lower()
        if t in ('vs', 'v', 'vs.'):
            continue
        if t in MAP_NAMES or (re.fullmatch(r'[a-z0-9_]+', t) and len(t) >= 3):
            return None, token

    return None, None

def get_match_winner_team_name(df_rounds: pd.DataFrame):
    """
    Determines the winner and defeated team names of the match based on the most rounds won.
    Returns a tuple: (winner_team_name, defeated_team_name, winner_score, defeated_score),
    or (None, None, None, None) if no winner can be determined.
    """
    if 'winner_clan_name' not in df_rounds.columns:
        raise ValueError("df_rounds must have a 'winner_clan_name' column.")
    # Count the number of rounds won by each team
    round_wins = df_rounds['winner_clan_name'].value_counts()
    if round_wins.empty or len(round_wins) < 2:
        return (None, None, None, None)
    winner_team = round_wins.idxmax()
    winner_score = round_wins.max()
    defeated_team = round_wins.index[1] if round_wins.index[0] == winner_team else round_wins.index[0]
    defeated_score = round_wins[defeated_team]
    return (winner_team, defeated_team, winner_score, defeated_score)

def get_file_modified_date(file_path: str) -> str:
    """
    Returns the last modified date of the file as a string in 'YYYY-MM-DD HH:MM:SS' format.
    """
    timestamp = os.path.getmtime(file_path)
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

def is_valid_dem(file_path: str, min_size: int = 1024) -> bool:
    """Quick sanity checks for a .dem file to avoid passing clearly truncated/corrupt files to the parser.

    Checks performed:
    - file size is above a small threshold (default 1KB)
    - the file header contains the expected magic bytes ('HL2DEMO') used by Source engine demo files

    This is a lightweight heuristic (not a full validation). If it returns False the script will skip the file
    instead of letting the parser raise OutOfBytesError.
    """
    try:
        if not os.path.isfile(file_path):
            return False
        size = os.path.getsize(file_path)
        if size < min_size:
            return False
        with open(file_path, 'rb') as fh:
            header = fh.read(16)
        # Most Source engine demos start with the ASCII 'HL2DEMO' magic string
        if b'HL2DEMO' in header:
            return True
        # Some variants may have slightly different header layouts; fall back to size-only check
        return size >= min_size
    except Exception:
        return False

def convert_to_nullable_int(df, columns=None):
    """Convert specified columns to nullable Int64, handling NaN values."""
    if columns is None:
        # Auto-detect numeric columns with NaN
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(pd.NA).astype('Int64')
    
    return df
