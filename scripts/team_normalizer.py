"""
Team Name Normalizer

Handles team name variations from demo files to ensure consistent team identification
across matches. Uses static mapping for known teams and fuzzy matching for variations.
"""

import pandas as pd
from difflib import SequenceMatcher
from typing import Tuple, List, Dict
import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# CANONICAL TEAM MAPPING
# Built from d_teams.csv - maps all known variations to canonical team names
# ============================================================================

TEAM_MAPPING = {
    # 3DMAX
    "TEAM 3DMAX": "3DMAX",
    
    # Astralis
    # No variations known
    
    # FURIA
    # No variations known
    
    # FlyQuest
    "FLYQUEST": "FlyQuest",
    
    # GamerLegion
    # No variations known
    
    # Team Spirit
    "Spirit": "Team Spirit",
    
    # The MongolZ
    "The Mongolz": "The MongolZ",
    
    # Team Liquid
    "Liquid": "Team Liquid",
    
    # MOUZ
    # No variations known
    
    # Natus Vincere
    "Natus Vincere NAVI": "Natus Vincere",
    
    # Team Vitality
    "Vitality": "Team Vitality",
    "Team Vitality CS2": "Team Vitality",
    
    # Team Falcons
    "Falcons": "Team Falcons",
    
    # FaZe Clan
    "FaZe": "FaZe Clan",
    "FaZe Clan CS2": "FaZe Clan",
    
    # G2 Esports
    "G2": "G2 Esports",
    "G2 Esports CS2": "G2 Esports",
    
    # BIG
    "BIG CS2": "BIG",
    
    # Eternal Fire
    # No variations known
    
    # MIBR
    "Made in Brazil": "MIBR",
    
    # SAW
    # No variations known
    
    # Complexity
    "ComplexityCS": "Complexity",
    
    # paiN Gaming
    "paiN": "paiN Gaming",
    
    # LEGACY
    "Legacy_": "LEGACY",
    "Legacy": "LEGACY",
    
    # Betclic Apogee
    # No variations known
    
    # Aurora Gaming
    "Aurora": "Aurora Gaming",
    "Aurora Gaming -": "Aurora Gaming",
    
    # ATOX
    # No variations known
    
    # ENCE
    "ENCE CS2": "ENCE",
    
    # ECSTATIC
    # No variations known
    
    # Imperial Esports
    "Imperial": "Imperial Esports",
    "Imperial Sportsbet": "Imperial Esports",
    "Gamdom Imperial": "Imperial Esports",
    
    # Fnatic
    # No variations known
    
    # HEROIC
    # No variations known
    
    # Lynn Vision Gaming
    "Lynn Vision": "Lynn Vision Gaming",
    "Lynn Vsion Gaming": "Lynn Vision Gaming",  # Typo in original data
    
    # M80
    "M80 CS2": "M80",
    
    # Nemiga
    # No variations known
    
    # NRG
    "NRG.GG": "NRG",
    
    # Ninjas in Pyjamas
    # No variations known
    
    # ODDIK
    # No variations known
    
    # OG Esports
    "OG CS2": "OG Esports",
    
    # PARIVISION
    # No variations known
    
    # Passion UA
    # No variations known
    
    # Rare Atom
    "RARE ATOM": "Rare Atom",
    "RareAtomCS": "Rare Atom",
    
    # RED Canids
    # No variations known
    
    # Rooster
    # No variations known
    
    # Sangal 1XBET
    # No variations known
    
    # Movistar KOI
    # No variations known
    
    # Tyloo
    "TYLOO": "Tyloo",
    
    # Virtus.Pro
    "Virtus.pro": "Virtus.Pro",
    "V.P.": "Virtus.Pro",
    
    # Wildcard Gaming
    "Wildcard": "Wildcard Gaming",
    
    # BetBoom Team
    # No variations known
    
    # Clutches (if appears)
    # No variations known
    
    # The Huns
    "HUNS": "The Huns",
    "THE HUNS": "The Huns",
    
    # BCG
    # No variations known
    
    # FUT Esports
    "FUT": "FUT Esports",
    
    # Fluxo
    # No variations known
    
    # Gentle Mates
    # "Gentle mates": "Gentle Mates",  # Case variation
    
    # HOTU Esports
    "HOTU": "HOTU Esports",
    
    # B8 Esports
    "B8": "B8 Esports",
    "[B8]": "B8 Esports",
    
    # 9INE
    # No variations known
    
    # 9z Globant
    # No variations known
}

# ============================================================================
# NORMALIZATION TRACKING
# ============================================================================

normalization_log: List[Dict] = []


def normalize_team_name(raw_name: str) -> Tuple[str, str]:
    """
    Normalize a raw team name to its canonical form.
    
    Process:
    1. Check static mapping (known variations)
    2. Try fuzzy match against canonical names (95%+ similarity)
    3. Treat as new team if no match found
    
    Args:
        raw_name: Raw team name from demo file
        
    Returns:
        Tuple of (canonical_name, normalization_type)
        normalization_type: "STATIC" | "FUZZY_XX%" | "NEW_TEAM"
        
    Example:
        >>> normalize_team_name("HUNS")
        ('The Huns', 'STATIC')
        
        >>> normalize_team_name("THEHUNS")
        ('The Huns', 'FUZZY_96%')
        
        >>> normalize_team_name("NewTeam2025")
        ('NewTeam2025', 'NEW_TEAM')
    """
    
    if not raw_name or pd.isna(raw_name):
        return None, "INVALID"
    
    raw_name = str(raw_name).strip()
    
    # Step 1: Check static mapping
    if raw_name in TEAM_MAPPING:
        canonical = TEAM_MAPPING[raw_name]
        log_normalization(raw_name, canonical, "STATIC")
        return canonical, "STATIC"
    
    # If raw name is already a canonical team name, return it
    if raw_name in set(TEAM_MAPPING.values()):
        log_normalization(raw_name, raw_name, "STATIC_CANONICAL")
        return raw_name, "STATIC_CANONICAL"
    
    # Step 2: Try fuzzy match
    best_match, confidence = fuzzy_match_team(
        raw_name,
        list(set(TEAM_MAPPING.values())),  # Use canonical names
        threshold=0.92
    )
    
    if best_match:
        norm_type = f"FUZZY_{confidence}%"
        log_normalization(raw_name, best_match, norm_type)
        return best_match, norm_type
    
    # Step 3: Treat as new team
    log_normalization(raw_name, raw_name, "NEW_TEAM")
    return raw_name, "NEW_TEAM"


def fuzzy_match_team(
    name: str,
    candidates: List[str],
    threshold: float = 0.95
) -> Tuple[str, int]:
    """
    Find best fuzzy match for a team name.
    
    Args:
        name: Raw team name to match
        candidates: List of canonical team names to match against
        threshold: Minimum similarity ratio (0.0-1.0), default 0.95 = 95%
        
    Returns:
        Tuple of (best_match, confidence_percentage) or (None, 0) if no match
    """
    
    best_match = None
    best_ratio = 0
    
    for candidate in candidates:
        ratio = SequenceMatcher(None, name.lower(), candidate.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = candidate
    
    if best_ratio >= threshold:
        confidence = int(round(best_ratio * 100))
        return best_match, confidence
    
    return None, 0


def log_normalization(raw_name: str, canonical_name: str, norm_type: str) -> None:
    """
    Log a team name normalization event.
    
    Args:
        raw_name: Raw team name from demo
        canonical_name: Canonical team name it was mapped to
        norm_type: Type of normalization applied
    """
    normalization_log.append({
        'raw_name': raw_name,
        'canonical_name': canonical_name,
        'normalization_type': norm_type,
        'timestamp': datetime.now().isoformat()
    })


def save_normalization_log(filepath: str) -> None:
    """
    Save normalization log to CSV file for review.
    
    Args:
        filepath: Path to save log CSV file
    """
    if not normalization_log:
        print("No normalizations logged.")
        return
    
    df = pd.DataFrame(normalization_log)
    df.to_csv(filepath, index=False)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"TEAM NORMALIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total normalizations: {len(df)}")
    print(f"\nBreakdown by type:")
    for norm_type, count in df['normalization_type'].value_counts().items():
        print(f"  {norm_type}: {count}")
    
    # Show new teams
    new_teams = df[df['normalization_type'] == 'NEW_TEAM']['canonical_name'].unique()
    if len(new_teams) > 0:
        print(f"\nNew teams found ({len(new_teams)}):")
        for team in sorted(new_teams):
            count = len(df[(df['canonical_name'] == team) & (df['normalization_type'] == 'NEW_TEAM')])
            print(f"  {team} (appeared {count} time(s))")
    
    # Show fuzzy matches
    fuzzy_matches = df[df['normalization_type'].str.contains('FUZZY')]
    if len(fuzzy_matches) > 0:
        print(f"\nFuzzy matches ({len(fuzzy_matches.groupby('raw_name'))}):")
        for raw, group in fuzzy_matches.groupby('raw_name'):
            canonical = group['canonical_name'].iloc[0]
            norm_type = group['normalization_type'].iloc[0]
            count = len(group)
            print(f"  {raw} → {canonical} ({norm_type}, appeared {count} time(s))")
    
    print(f"\nLog saved to: {filepath}")
    print(f"{'='*70}\n")


def clear_log() -> None:
    """Clear the normalization log."""
    global normalization_log
    normalization_log = []


def get_normalization_stats() -> Dict:
    """
    Get statistics about normalizations performed.
    
    Returns:
        Dictionary with stats
    """
    if not normalization_log:
        return {}
    
    df = pd.DataFrame(normalization_log)
    
    return {
        'total_normalizations': len(df),
        'static_mappings': len(df[df['normalization_type'] == 'STATIC']),
        'fuzzy_matches': len(df[df['normalization_type'].str.contains('FUZZY')]),
        'new_teams': len(df[df['normalization_type'] == 'NEW_TEAM']),
        'unique_new_teams': len(df[df['normalization_type'] == 'NEW_TEAM']['canonical_name'].unique()),
        'normalization_types': df['normalization_type'].value_counts().to_dict()
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_all_known_teams() -> List[str]:
    """Get list of all known canonical team names."""
    return sorted(list(set(TEAM_MAPPING.values())))


def add_custom_mapping(raw_name: str, canonical_name: str) -> None:
    """
    Add a custom team name mapping at runtime.
    
    Args:
        raw_name: Raw team name to add
        canonical_name: Canonical name it should map to
    """
    TEAM_MAPPING[raw_name] = canonical_name
    print(f"Added mapping: '{raw_name}' → '{canonical_name}'")
    