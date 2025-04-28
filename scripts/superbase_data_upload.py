# Imports
import pandas as pd
import numpy as np
import os
from supabase import create_client, Client
from dotenv import load_dotenv
from requests import post, get

# Load environment variables from .env file
load_dotenv()
url = os.getenv("url")
key = os.getenv("key")
supabase: Client = create_client(url, key)

event_name = "IEM_Katowice_2025"

def insert_table(df, table_name):
    for row in df.to_dict(orient="records"):
        supabase.table(table_name).insert(row).execute()

player_stats = pd.read_csv(f"data_export_{event_name}.csv")
player_stats = player_stats.drop(columns=["Unnamed: 0"])
players_df = player_stats[['steam_id', 'user_name', 'team_clan_name']].drop_duplicates().reset_index(drop=True)

teams = pd.read_csv(f"teams_{event_name}.csv")
teams = teams['team_clan_name'].dropna().unique().tolist()

rounds = pd.read_csv(f"rounds_{event_name}.csv")
rounds = rounds.drop(columns=["Unnamed: 0"])
rounds['bomb_plant'] = rounds['bomb_plant'].replace({np.nan: None})
rounds['bomb_site'] = rounds['bomb_site'].replace({np.nan: None})

matches = pd.read_csv(f"matches_{event_name}.csv")
matches = matches[["match_name", "event_id"]].to_dict("records")

player_match_summary = pd.read_csv(f"player_match_summary_{event_name}.csv")

kill_stats_cols = [
    'steam_id', 'kills', 'headshots', 'wallbang_kills', 'no_scope',
    'through_smoke', 'airbone_kills', 'blind_kills', 'victim_blind_kills',
    'awp_kills', 'pistol_kills', 'first_kills', 'ct_first_kills', 't_first_kills',
    'first_deaths', 'ct_first_deaths', 't_first_deaths'
]
kill_stats_df = player_stats[kill_stats_cols]

general_stats_cols = [
    'steam_id', 'assists', 'deaths', 'trade_kills', 'trade_deaths', 'kd', 'k_d_diff',
    'adr_total', 'adr_ct_side', 'adr_t_side', 'kast_total', 'kast_ct_side',
    'kast_t_side', 'total_rounds_won', 't_rounds_won', 'ct_rounds_won'
]
general_stats_df = player_stats[general_stats_cols]

utility_stats_cols = [
    'steam_id', 'assisted_flashes', 'flahes_thrown', 'ct_flahes_thrown', 't_flahes_thrown',
    'flahes_thrown_in_pistol_round', 'he_thrown', 'ct_he_thrown', 't_he_thrown',
    'he_thrown_in_pistol_round', 'infernos_thrown', 'ct_infernos_thrown', 't_infernos_thrown',
    'infernos_thrown_in_pistol_round', 'smokes_thrown', 'ct_smokes_thrown', 't_smokes_thrown',
    'smokes_thrown_in_pistol_round', 'util_in_pistol_round', 'total_util_thrown', 'total_util_dmg', 'ct_total_util_dmg', 't_total_util_dmg'
]
utility_stats_df = player_stats[utility_stats_cols]

# Insert teams_data into the database
teams_data = [{"team_clan_name": name} for name in teams]
supabase.table("teams").upsert(teams_data, on_conflict="team_clan_name").execute()

# Gets the team_id created by supabase
response = supabase.table("teams").select("id, team_clan_name").execute()
team_id_map = {item['team_clan_name']: item['id'] for item in response.data}
players_df["team_id"] = players_df["team_clan_name"].map(team_id_map)
players_df = players_df[["steam_id", "user_name", "team_id"]].drop_duplicates().to_dict("records")

# Insert players_data into the database
response = supabase.table("players").upsert(players_df, on_conflict="steam_id").execute()

# Insert teams_data into the database
supabase.table("matches").upsert(matches, on_conflict="match_name, event_id").execute()

# Gets the team_id created by supabase
response = supabase.table("matches").select("file_id, match_name", count="exact").execute()
file_id_map = {item['match_name']: item['file_id'] for item in response.data}
rounds["file_id"] = rounds["match_name"].map(file_id_map)
print(rounds)
# # Other tables
# insert_table(kill_stats_df, "kill_stats")
# insert_table(general_stats_df, "general_stats")
# insert_table(utility_stats_df, "utility_stats")
# insert_table(matches, "matches")
# insert_table(rounds, "rounds")
# insert_table(player_match_summary, "player_match_summary")