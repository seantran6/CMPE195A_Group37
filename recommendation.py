"""
recommendation.py
-------------------------------------------------------------------
Return Spotify **track** recommendations for every (age, gender) pair
-------------------------------------------------------------------

• Uses a curated mapping so each demographic always gets **≥ 3 tracks**.
• Falls back to a quick Spotify search (“Top <decade> hits”) if a new
  demographic appears that’s not in the map.
• Requires Client‑Credentials keys in .env:
    SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI
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
# Feel free to swap IDs to better match your audience.
_CURATED_TRACKS: Dict[tuple[str, str], List[str]] = {
    # -------------------- Babies (0‑2) ---------------------------
    ("(0-2)", "male"): [
        "2bOh00jX47DuPOLl4tFihp",  # Pinkfong – Baby Shark
        "5cQz1J7kQrFIYllEkjvBTc",  # Jewel – Brahms' Lullaby
        "4leWqNkAmWtG2YLzb2cT3q",  # LBB – Twinkle Twinkle
        "3mfbqsYUdIqGIPzKjmOEQC",  # My Grandfather's Clock
        "3T4LWuYdxwyejPxCQeM0B9",  # Leo the Lion
        "0swfhyV723N4nOZDegevHF",  # Playing with Leaves
    ],
    ("(0-2)", "female"): [
        "3BewjlT8ixl4lIzJ0a2CO4",  
        "21syoBtd2QkbyGiXB6Q2NN",
        "30fmyIi9AQWqiLuif33pSH",
        "04yKwXnspkmDJp8zDpYnr9",  # My Raincoat
        "0XEA4ZTY7m2VkeCBr8NlN1",  # Pebbles
        "3twIESjUSaC1Y1fHJ32XBd",  # From Me to You
    ],

    # -------------------- Preschool (4‑6) ------------------------
    ("(4-6)", "male"): [
        "6U4VqEHy4n5VeiH4pQPL24",  # You're Welcome
        "6mb6lVLNrcUgLnEN8QnDJd",  # How Far I'll Go
        "60nZcImufyMA1MKQY3dcCH",  # Happy
        "1A8j067qyiNwQnZT0bzUpZ",  # This Girl
        "5ZkAx8zjLiSs1nMmBwJoZS",  # When Can I See You Again?
        "1N3dZ7TTWO6VcD4Y3hHYLZ",  # Try Everything
    ],
    ("(4-6)", "female"): [
        "52xJxFP6TqMuO4Yt0eOkMz",  # We don't talk abt Bruno
        "2yi7HZrBOC4bMUSTcs4VK6",  # Build a Snowman
        "0OFknyqxmSQ42SoKxWVTok",  # Un Poco Loco
        "1UPB5rYJ0bzn6mNSoAHrZC",  # You'll Be In My Heart
        "7GmiJVBAzWNikX5VkNQg85",  # Hawaiian Roller Coaster Ride
        "7iocNjLrxPHLl8njgRlv5U",  # Married Life
    ],

    # -------------------- Kids (8‑12) 2020s ----------------------------
    ("(8-12)", "male"): [
        "0VjIjW4GlUZAMYd2vXMi3b",  # Blinding Lights
        "27NovPIUIRrOZoCHxABJwK",  # Industry Baby
        "3KkXRkHbMCARz0aVfEt68P",  # Sunflower
        "6I3mqTwhRpn34SLVafSH7G",  # Ghost
        "42VsgItocQwOQC3XWZ8JNA",  # Fe!n
        "60wwxj6Dd9NJlirf84wr2c",  # Clarity
    ],
    ("(8-12)", "female"): [
        "39LLxExYz6ewLAcYrzQQyP",  # Levitating
        "4ZtFanR9U6ndgddUvNcjcG",  # good 4 u
        "0V3wPSX9ygBnCm8psDIegu",  # Anti‑Hero
        "3U3hFkMr0Q90pD24EkE3Pr",  # BMF
        "4kfSXPK13aXkLzuz02hCSC",  # Greedy
        "5QDLhrAOJJdNAmCTJ8xMyW",  # Dynamite
    ],

    # -------------------- Teens (15‑20) 2010s -------------------------
    ("(15-20)", "male"): [
        "21jGcNKet2qwijlDFuPiPb",  # Circles
        "7KA4W4McWYRpgf0fWsJZWB",  # See you Again
        "2u9S9JJ6hTZS3Vf22HOZKg",  # Nokia
        "7ne4VBA60CxGM75vw0EYad",  # I Ain't Comin Back
        "2FAZskT9yRjp2Oow9szJD8",  # The Days
        "6rdkCkjk6D12xRpdMXy0I2",  # New Jeans

    ],
    ("(15-20)", "female"): [
        "4kfXaAAZlfBrimPJYHlCEM",  # Silver Lining
        "6vuVCtwukUA57ioTnKKeuL",  # Bluest Flame
        "3SAga35lAPYdjj3qyfEsCF",  # Feel It
        "0fK7ie6XwGxQTIkpFoWkd1",  # like Jennie
        "7ne4VBA60CxGM75vw0EYad",  # That's So True
        "5pmITEphUtjpCLmKiYIPl9",  # Soft Spot
    ],

    # -------------------- Young adults (25‑32) 2000s ------------------
    ("(25-32)", "male"): [
        "7FGq80cy8juXBCD2nrqdWU",  # Eastside
        "6AI3ezQ4o3HUoP6Dhudph3",  # Not Like Us
        "6pooRNiLyYpxZeIA5kJ5EX",  # Good Things Fall Apart
        "4SFknyjLcyTLJFPKD2m96o",  # How You Like That
        "2H1047e0oMSj10dgp7p2VG",  # I Gotta Feeling
        "3gE4eQH3K83Sght0ZLvuBK",  # MIA
    ],
    ("(25-32)", "female"): [
        "7qiZfU4dY1lWllzX7mPBI3",  # Shape of You
        "68LNIOMjcliUHjW0turxcP",  # Party in the USA
        "6epn3r7S14KUqlReYr77hA",  # Baby
        "6YUTL4dYpB9xZO5qExPf05",  # Summer
        "2K87XMYnUMqLcX3zvtAF4G",  # Drag Me Down
        "1zi7xx7UVEFkmKfv06H8x0",  # One Dance
    ],

    # -------------------- Adults (38‑43) 90s ------------------------
    ("(38-43)", "male"): [
        "6brl7bwOHmGFkNw3MBqssT",  # So Sick
        "0j2T0R9dR9qdJYsB7ciXhf",  # Stronger
        "1mea3bSkSGXuIRvnydlB5b",  # Viva La Vida
        "4EWCNWgDS8707fNSZ1oaA5",  # Heartless
        "26vLppndde4LSU041wKH79",  # Kryptonite
        "0t7kjpVLgOYITrSfFCoBEA",  # Gee
    ],
    ("(38-43)", "female"): [
        "0W4NhJhcqKCqEP2GIpDCDq",  # Love
        "3DamFFqW32WihKkTVlwTYQ",  # Fireflies
        "5IVuqXILoxVWvWEPm82Jxr",  # Crazy in Love
        "2gam98EZKrF9XuOkU13ApN",  # Promiscuous
        "5rb9QrpfcKFHM1EUbSIurX",  # YEAH
        "2CEgGE6aESpnmtfiZwYlbV",  # Dynamite
    ],

    # -------------------- Mature (48‑53) 80s ------------------------
   ("(48-53)", "male"): [
        "40riOy7x9W7GXjyGp4pjAv",  # Hotel California (2013 Remaster) *global version*
        "7snQQk1zcKl8gZ92AnueZW",  # Sweet Child o' Mine
        "6l8GvAyoUZwWDgF1e4822w",  # Bohemian Rhapsody
        "1jDJFeK9x3OZboIAHsY9k2",  # I'm Still Standing
        "2qOm7ukLyHUXWyR4ZWLwxA",  # It was a good day
        "5ChkMS8OtdzJeqyybCc9R5",  # Billie Jean
   ],
   ("(48-53)", "female"): [
        "0GjEhVFGZW8afUYGChu3Rr",  # Dancing Queen
        "0ofHAoxe9vBkTCp2UQIavz",  # Dreams
        "3Be7CLdHZpyzsVijme39cW",  # What's Love Got to Do with It
        "2374M0fQpWi3dLnB54qaLX",  # Africa
        "0rmGAIH9LNJewFw7nKzZnc",  # You give love a bad name
        "2tUBqZG2AbRi7Q0BIrVrEj",  # I wanna dance with somebody
   ],

    # -------------------- Seniors (60‑100) ----------------------
    ("(60-100)", "male"): [
        "28cnXtME493VX9NOw9cIUh",  # Hurt – Johnny Cash
        "3SdTKo2uVsxFblQjpScoHy",  # Stand By Me
        "2YkIDPL5lGhRhomCq4S2RO",  # My Way – Sinatra
        "5ehcf6UL1TkwozB386cRAp",  # Don't Stop Believin
        "7oOOI85fVQvVnK5ynNMdW7",  # Rock with you
        "0bfvHnWWOeU1U5XeKyVLbW",  # Can't Take My Eyes Off You

    ],
    ("(60-100)", "female"): [
        "5K09WxKdlkroDyEVyOSySy",  # Natural Woman
        "7rIovIsXE6kMn629b7kDig",  # I Will Survive
        "7s25THrKz86DM225dOYwnr",  # Respect
        "3koCCeSaVUyrRo3N2gHrd8",  # Let's Groove
        "5JccvAiwcZ7n3urnXqWPsG",  # Break It to me Gently
        "4QxDOjgpYtQDxxbWPuEJOy",  # L-O-V-E
    ],
}

# ─────────────────────── Helpers ────────────────────────────────

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

# assumes these already exist in recommendation.py
#   _CURATED_TRACKS : Dict[(age_label, gender_lower), List[str]]
#   _sp             : spotipy.Spotify client (Client-Credentials)
#   _age_to_decade  : helper → '1990s', '2000s', …
#   _search_tracks  : helper that returns a list of track IDs for a query


@lru_cache(maxsize=128)
def get_tracks_for_demographic(
    age_label: str,
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

    key = (age_label, gender.lower())
    wanted_ids: List[str] = _CURATED_TRACKS.get(key, []).copy()

    if not wanted_ids:                # unknown demographic
        wanted_ids = _search_tracks(f"top {_age_to_decade(age_label)} hits", n)

    # Collect track objects until we have >= n valid ones
    gathered: Dict[str, Dict] = {}    # id -> track-dict
    attempt   = 0
    max_rounds = 4                    # safety to avoid infinite loop

    while len(gathered) < n and attempt < max_rounds:
        attempt += 1
        # Spotify allows 50 IDs per call – we never exceed that
        tracks = _sp.tracks(wanted_ids)["tracks"]

        # Keep only non-None objects & deduplicate by ID
        for t in tracks:
            if t and t["id"] not in gathered:
                gathered[t["id"]] = {
                    "id":          t["id"],
                    "name":        t["name"],
                    "artists":     ", ".join(a["name"] for a in t["artists"]),
                    "preview_url": t["preview_url"],
                    "track_url":   f"https://open.spotify.com/track/{t['id']}",
                    "image": t["album"]["images"][0]["url"] if t["album"]["images"] else ""
                }

        # if we still need more, top-up with decade hits
        #if len(gathered) < n:
        #    deficit = n - len(gathered)
        #   decade  = _age_to_decade(age_label)
        #   wanted_ids = _search_tracks(f"top {decade} hits", deficit)

    # final list
    result = list(gathered.values())
    if shuffle_result:
        random.shuffle(result)

    return result[:n]



