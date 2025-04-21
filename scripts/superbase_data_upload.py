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

df = pd.DataFrame({
    "nome": ["Ana", "João", "Maria"],
    "idade": [25, 30, 28]
})

for row in df.to_dict(orient="records"):
    response = supabase.table("my_table").insert(row).execute()
    print(response)