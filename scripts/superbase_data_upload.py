# Imports
import pandas as pd
import os
from supabase import create_client, Client
from dotenv import load_dotenv
from requests import post, get

# Load environment variables from .env file
load_dotenv()
url = os.getenv("url")
key = os.getenv("key")

supabase: Client = create_client(url, key)

df = pd.read_csv("Data_Export.csv")
df = df.drop(columns=["Unnamed: 0"])

teams_df = df[['team_clan_name']].drop_duplicates().reset_index(drop=True)

players_df = df[['steam_id', 'user_name', 'team_clan_name']].drop_duplicates().reset_index(drop=True)

kill_stats_cols = [
    'steam_id', 'kills', 'headshots', 'wallbang_kills', 'no_scope',
    'through_smoke', 'airbone_kills', 'blind_kills', 'victim_blind_kills',
    'awp_kills', 'pistol_kills', 'first_kills', 'ct_first_kills', 't_first_kills',
    'first_deaths', 'ct_first_deaths', 't_first_deaths'
]
kill_stats_df = df[kill_stats_cols]

general_stats_cols = [
    'steam_id', 'assists', 'deaths', 'trade_kills', 'trade_deaths', 'kd', 'k_d_diff',
    'adr_total', 'adr_ct_side', 'adr_t_side', 'kast_total', 'kast_ct_side',
    'kast_t_side', 'total_rounds_won', 't_rounds_won', 'ct_rounds_won'
]
general_stats_df = df[general_stats_cols]

utility_stats_cols = [
    'steam_id', 'assisted_flashes', 'flahes_thrown', 'ct_flahes_thrown', 't_flahes_thrown',
    'flahes_thrown_in_pistol_round', 'he_thrown', 'ct_he_thrown', 't_he_thrown',
    'he_thrown_in_pistol_round', 'infernos_thrown', 'ct_infernos_thrown', 't_infernos_thrown',
    'infernos_thrown_in_pistol_round', 'smokes_thrown', 'ct_smokes_thrown', 't_smokes_thrown',
    'smokes_thrown_in_pistol_round', 'util_in_pistol_round', 'total_util_thrown'
]
utility_stats_df = df[utility_stats_cols]

# Teams
team_ids = {}
for _, row in teams_df.iterrows():
    team_name = row['team_clan_name']
    result = supabase.table('teams').insert({'team_clan_name': team_name}).execute()
    team_ids[team_name] = result.data[0]['id']

# Players
for _, row in players_df.iterrows():
    supabase.table('players').insert({
        'steam_id': int(row['steam_id']),
        'user_name': row['user_name'],
        'team_id': team_ids[row['team_clan_name']]
    }).execute()
    
# Other tables
def insert_table(df, table_name):
    for row in df.to_dict(orient="records"):
        supabase.table(table_name).insert(row).execute()

insert_table(kill_stats_df, "kill_stats")
insert_table(general_stats_df, "general_stats")
insert_table(utility_stats_df, "utility_stats")