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

for row in df.to_dict(orient="records"):
    response = supabase.table("player_stats").insert(row).execute()
    # print(response)
print("Data uploaded successfully!")