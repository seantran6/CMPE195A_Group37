from typing import List, Dict

def get_recommendations(age: int, gender: str) -> List[Dict[str, str]]:
    # Determine age bucket based on age
    if age <= 2:
        age_bucket = "(0-2)"
    elif age <= 6:
        age_bucket = "(4-6)"
    elif age <= 12:
        age_bucket = "(8-12)"
    elif age <= 20:
        age_bucket = "(15-20)"
    elif age <= 32:
        age_bucket = "(25-32)"
    elif age <= 43:
        age_bucket = "(38-43)"
    elif age <= 53:
        age_bucket = "(48-53)"
    else:
        age_bucket = "(60-100)"

    # Hardcoded song recommendations with links
    recommendations = {
        "(0-2)": [
            {"title": "Twinkle Twinkle Little Star", "link": "https://www.youtube.com/watch?v=yCjJyiqpAuU"},
            {"title": "Baby Shark", "link": "https://www.youtube.com/watch?v=XqZsoesa55w"},
            {"title": "Wheels on the Bus", "link": "https://www.youtube.com/watch?v=GzrjwOQpAl0"}
        ],
        "(4-6)": [
            {"title": "Let It Go - Idina Menzel", "link": "https://www.youtube.com/watch?v=L0MK7qz13bU"},
            {"title": "Old MacDonald Had a Farm", "link": "https://www.youtube.com/watch?v=_6HzoUcx3eo"},
            {"title": "If You're Happy and You Know It", "link": "https://www.youtube.com/watch?v=71hqRT9U0wg"}
        ],
        "(8-12)": [
            {"title": "Happy - Pharrell Williams", "link": "https://www.youtube.com/watch?v=ZbZSe6N_BXs"},
            {"title": "Shake It Off - Taylor Swift", "link": "https://www.youtube.com/watch?v=nfWlot6h_JM"},
            {"title": "Roar - Katy Perry", "link": "https://www.youtube.com/watch?v=CevxZvSJLk8"}
        ],
        "(15-20)": [
            {"title": "Blinding Lights - The Weeknd", "link": "https://www.youtube.com/watch?v=4NRXx6U8ABQ"},
            {"title": "Shape of You - Ed Sheeran", "link": "https://www.youtube.com/watch?v=JGwWNGJdvx8"},
            {"title": "Uptown Funk - Mark Ronson ft. Bruno Mars", "link": "https://www.youtube.com/watch?v=OPf0YbXqDm0"}
        ],
        "(25-32)": [
            {"title": "Hey Ya! - Outkast", "link": "https://www.youtube.com/watch?v=PWgvGjAhvIw"},
            {"title": "Rolling in the Deep - Adele", "link": "https://www.youtube.com/watch?v=rYEDA3JcQqw"},
            {"title": "Viva La Vida - Coldplay", "link": "https://www.youtube.com/watch?v=dvgZkm1xWPE"}
        ],
        "(38-43)": [
            {"title": "Smells Like Teen Spirit - Nirvana", "link": "https://www.youtube.com/watch?v=hTWKbfoikeg"},
            {"title": "Wonderwall - Oasis", "link": "https://www.youtube.com/watch?v=6hzrDeceEKc"},
            {"title": "Losing My Religion - R.E.M.", "link": "https://www.youtube.com/watch?v=xwtdhWltSIg"}
        ],
        "(48-53)": [
            {"title": "Billie Jean - Michael Jackson", "link": "https://www.youtube.com/watch?v=Zi_XLOBDo_Y"},
            {"title": "Sweet Child O' Mine - Guns N' Roses", "link": "https://www.youtube.com/watch?v=1w7OgIMMRc4"},
            {"title": "Livin' on a Prayer - Bon Jovi", "link": "https://www.youtube.com/watch?v=lDK9QqIzhwk"}
        ],
        "(60-100)": [
            {"title": "Yesterday - The Beatles", "link": "https://www.youtube.com/watch?v=NrgmdOz227I"},
            {"title": "My Girl - The Temptations", "link": "https://www.youtube.com/watch?v=6IUG-9jZD-g"},
            {"title": "What a Wonderful World - Louis Armstrong", "link": "https://www.youtube.com/watch?v=CWzrABouyeE"}
        ]
    }

    # Return the song list for the detected age bucket; gender ignored for now
    return recommendations.get(age_bucket, [{"title": "Let It Be - The Beatles", "link": "https://www.youtube.com/watch?v=QDYfEBY9NM4"}])

# Example usage:
if __name__ == "__main__":
    age = 30
    gender = "male"
    recs = get_recommendations(age, gender)
    for song in recs:
        print(f"{song['title']}: {song['link']}")

get_tracks_for_demographic = get_recommendations
