# recommendation.py

def get_tracks_for_demographic(age, gender, n=6):
    try:
        age = int(age)
    except (ValueError, TypeError):
        age_range = "unknown"
    else:
        if age < 13:
            age_range = "child"
        elif age < 20:
            age_range = "teen"
        elif age < 30:
            age_range = "young_adult"
        elif age < 45:
            age_range = "adult"
        elif age < 65:
            age_range = "middle_aged"
        else:
            age_range = "senior"

    return select_tracks_by_demographic(age_range, gender, n)


def select_tracks_by_demographic(age_range, gender, n=6):
    # Dummy example playlist (use real Spotify URIs or metadata in production)
    playlist = {
        ("child", "Male"): ["child_male_1", "child_male_2"],
        ("child", "Female"): ["child_female_1", "child_female_2"],
        ("teen", "Male"): ["teen_male_1", "teen_male_2"],
        ("teen", "Female"): ["teen_female_1", "teen_female_2"],
        ("young_adult", "Male"): ["ya_male_1", "ya_male_2"],
        ("young_adult", "Female"): ["ya_female_1", "ya_female_2"],
        ("adult", "Male"): ["adult_male_1", "adult_male_2"],
        ("adult", "Female"): ["adult_female_1", "adult_female_2"],
        ("middle_aged", "Male"): ["middle_male_1", "middle_male_2"],
        ("middle_aged", "Female"): ["middle_female_1", "middle_female_2"],
        ("senior", "Male"): ["senior_male_1", "senior_male_2"],
        ("senior", "Female"): ["senior_female_1", "senior_female_2"],
    }

    key = (age_range, gender)
    if key in playlist:
        return playlist[key][:n]
    else:
        return ["default_track_1", "default_track_2", "default_track_3"][:n]
