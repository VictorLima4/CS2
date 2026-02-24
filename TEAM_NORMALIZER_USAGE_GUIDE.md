# Team Normalizer - Database Sync Usage Guide

## Overview
The team_normalizer module now includes database synchronization capabilities to automatically keep the TEAM_MAPPING updated with all teams in your Supabase database.

## Quick Start

### Basic Usage - Sync without Variations

Add this after your Supabase client is created in `main_script_2.py`:

```python
from team_normalizer import sync_teams_from_database

# After creating the supabase client:
supabase: Client = create_client(url, key)

# Sync teams from database
result = sync_teams_from_database(supabase, schema_name="public")
```

### Advanced Usage - Sync with Naming Variations (Recommended)

For better team matching, use the enhanced sync function that automatically creates variations:

```python
from team_normalizer import sync_teams_with_variations

# After creating the supabase client:
supabase: Client = create_client(url, key)

# Sync teams with automatic naming variations
result = sync_teams_with_variations(supabase, schema_name="public", add_variations=True)
```

This creates variations like:
- Original: `FURIA`
- Lowercase: `furia`
- Uppercase: `FURIA` (same)
- Title case: `Furia`

## Functions Available

### 1. `sync_teams_from_database(supabase, schema_name="public")`

**Purpose:** Fetch all teams from the database and add any new teams to TEAM_MAPPING.

**Returns:**
```python
{
    'total_in_db': 50,              # Total teams found in database
    'already_mapped': 48,            # Teams already in TEAM_MAPPING
    'newly_added': ['NewTeam1', 'NewTeam2'],  # List of new teams added
    'count_added': 2                 # Number of teams added
}
```

**Example:**
```python
result = sync_teams_from_database(supabase)
print(f"Added {result['count_added']} new teams to mapping")
```

### 2. `sync_teams_with_variations(supabase, schema_name="public", add_variations=True)`

**Purpose:** Enhanced version that also adds naming variations for new teams.

**Returns:**
```python
{
    'total_in_db': 50,
    'already_mapped': 48,
    'newly_added': ['NewTeam1', 'NewTeam2'],
    'count_added': 2,
    'variations_added': 6            # Number of variation mappings added
}
```

**Example:**
```python
result = sync_teams_with_variations(supabase, add_variations=True)
print(f"Added {result['count_added']} teams with {result['variations_added']} variations")
```

### 3. `generate_naming_variations(team_name)`

**Purpose:** Generate common naming variations for a team name.

**Example:**
```python
from team_normalizer import generate_naming_variations

variations = generate_naming_variations("FURIA")
# Returns: ['FURIA', 'furia', 'Furia']
```

### 4. `get_db_team_names(supabase, schema_name="public")`

**Purpose:** Fetch all team names from the database without modifying TEAM_MAPPING.

**Example:**
```python
from team_normalizer import get_db_team_names

teams = get_db_team_names(supabase)
# Returns: ['3DMAX', 'Alliance', 'FURIA', ...]
```

## Integration Example

Here's how to integrate into your `main_script_2.py`:

```python
# ... existing imports ...
from supabase import create_client, Client
from dotenv import load_dotenv
from team_normalizer import sync_teams_with_variations

# ... existing code ...

# Database Management and Insertion
load_dotenv()
url = os.getenv("url")
key = os.getenv("key")
supabase: Client = create_client(url, key)

# ✅ NEW: Sync teams from database before processing
print("Syncing teams with database...")
sync_result = sync_teams_with_variations(supabase, add_variations=True)
print(f"Database sync complete: {sync_result['count_added']} new teams added")

# ... rest of your existing code ...
```

## How It Works

1. **Database Fetch**: Queries the `teams` table and retrieves all `team_clan_name` values
2. **Comparison**: Checks which teams are not already in TEAM_MAPPING (as either keys or values)
3. **Addition**: Adds new teams to TEAM_MAPPING with the database name as both the key and canonical value
4. **Variations** (optional): Generates common naming variations for better matching
5. **Summary**: Prints a summary of what was added

## Benefits

✅ Automatic team discovery - No manual updates needed when new teams are added  
✅ Consistency - All teams in the database are recognized during normalization  
✅ Flexibility - Naming variations improve fuzzy matching  
✅ Non-destructive - Existing mappings are never modified  
✅ Safe - Error handling prevents crashes on database issues  

## When to Sync

- **Startup**: Call sync function at the beginning of your main script
- **Periodic**: Call it during routine data processing if teams are frequently added
- **Debug**: Use `get_db_team_names()` to check what's in the database

## Notes

- The sync function uses the same Supabase schema as the rest of your code
- New teams are added with themselves as their canonical name (e.g., "NewTeam" → "NewTeam")
  - You can later update specific mappings using `add_custom_mapping()` if variations are discovered
- The function is safe to call multiple times - it only adds teams not already present
- Variations are only added for newly discovered teams, not existing ones

## Troubleshooting

### Function not found
Make sure you have the latest version of `team_normalizer.py` with the database sync functions.

### None of the new teams are being added
Check that:
1. Your Supabase client is properly initialized
2. The schema name matches your database (usually "public")
3. The teams table has data

### Want to add custom variations later?
Use the `add_custom_mapping()` function:
```python
from team_normalizer import add_custom_mapping

add_custom_mapping("FURIA Esports", "FURIA")
add_custom_mapping("Furia_Gaming", "FURIA")
```
