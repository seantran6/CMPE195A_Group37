import os
import random
from functools import lru_cache
from typing import List, Dict
from curated_tracks import _CURATED_TRACKS as CURATED_TRACKS

from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ─────────────────────── Spotify Client Setup ──────────────────────────
load_dotenv()

client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
print(client_id, client_secret)  # Debug print to confirm they're loaded

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
    )
)


# ─────────────────────── Curated Recommendations ──────────────────────
# Ensure at least 3 tracks per (age, gender) combination
from curated_tracks import _CURATED_TRACKS as CURATED_TRACKS

# ─────────────────────── Helpers ──────────────────────────────────────

def age_to_range(age: int) -> str | None:
    """Convert numeric age to age range string used in CURATED_TRACKS keys."""
    if 0 <= age <= 2:
        return "(0-2)"
    elif 4 <= age <= 6:
        return "(4-6)"
    elif 8 <= age <= 12:
        return "(8-12)"
    elif 15 <= age <= 20:
        return "(15-20)"
    elif 25 <= age <= 32:
        return "(25-32)"
    elif 38 <= age <= 43:
        return "(38-43)"
    elif 48 <= age <= 53:
        return "(48-53)"
    elif 60 <= age <= 100:
        return "(60-100)"
    else:
        return None  # age outside known ranges

def age_to_decade(age_label: str) -> str:
    try:
        low, high = map(int, age_label.strip("()").split("-"))
        midpoint = (low + high) // 2
        decade_start = (midpoint // 10) * 10
        decade_label = f"({decade_start}-{decade_start + 9})"
        return decade_label
    except Exception as e:
        raise ValueError(f"Invalid age label format: {age_label}") from e


def search_tracks(query: str, n: int) -> List[str]:
    try:
        res = sp.search(q=query, type="track", limit=n)["tracks"]["items"]
        return [track["id"] for track in res if track]
    except Exception as e:
        print(f"Spotify search failed for query '{query}': {e}")
        return []

# ─────────────────────── Public API ───────────────────────────────────

@lru_cache(maxsize=128)
def get_tracks_for_demographic(age_label: str, gender: str, n: int = 3, *, shuffle_result: bool = False) -> List[Dict]:
    """
    Returns a list of track dictionaries suitable for the given age and gender.
    If curated results are not available, it falls back to Spotify search.
    """
    key = (age_label, gender.lower())
    wanted_ids: List[str] = CURATED_TRACKS.get(key, []).copy()

    if not wanted_ids:
        decade = age_to_decade(age_label)
        wanted_ids = search_tracks(f"Top {decade} hits", n)

    gathered: Dict[str, Dict] = {}
    attempt = 0
    max_rounds = 4

    while len(gathered) < n and attempt < max_rounds:
        attempt += 1
        tracks = sp.tracks(wanted_ids)["tracks"]

        for t in tracks:
            if t and t["id"] not in gathered:
                gathered[t["id"]] = {
                    "id": t["id"],
                    "name": t["name"],
                    "artists": ", ".join(a["name"] for a in t["artists"]),
                    "preview_url": t["preview_url"],
                    "track_url": f"https://open.spotify.com/track/{t['id']}",
                    "image": t["album"]["images"][0]["url"] if t["album"]["images"] else ""
                }

        if len(gathered) < n:
            decade = age_to_decade(age_label)
            wanted_ids.extend(search_tracks(f"Top {decade} hits", n - len(gathered)))

    result = list(gathered.values())
    if shuffle_result:
        random.shuffle(result)

    return result[:n]

def get_tracks_for_age_and_gender(age: int, gender: str, n: int = 3, shuffle_result: bool = False) -> List[Dict]:
    """
    Wrapper function to convert numeric age to age range and get curated tracks.
    """
    age_range = age_to_range(age)
    if age_range is None:
        # Could return empty or fallback here, depending on design
        print(f"Warning: Age {age} not in predefined ranges. Returning empty list.")
        return []
    return get_tracks_for_demographic(age_range, gender, n, shuffle_result=shuffle_result)
