"""
Utility functions for DJ Mixing Recommendation System.
Includes Camelot Wheel conversion, key compatibility, and data loading helpers.
"""

import pandas as pd
import numpy as np
from difflib import SequenceMatcher


def key_to_camelot(key, mode):
    """
    Convert Spotify key (0-11) and mode (0=minor, 1=major) to Camelot Wheel notation.
    
    Args:
        key: Integer 0-11 (C=0, C#=1, D=2, D#=3, E=4, F=5, F#=6, G=7, G#=8, A=9, A#=10, B=11)
        mode: Integer 0 (minor) or 1 (major)
    
    Returns:
        String in Camelot notation (e.g., "8A", "9B")
    """
    # Camelot Wheel mapping: Spotify key + 1 (C=1, C#=2, ..., B=12)
    camelot_number = (key + 1) % 12
    if camelot_number == 0:
        camelot_number = 12
    
    # Mode: 0 (minor) -> B, 1 (major) -> A
    camelot_letter = "A" if mode == 1 else "B"
    
    return f"{camelot_number}{camelot_letter}"


def get_compatible_keys(camelot_key):
    """
    Get list of compatible Camelot keys for harmonic mixing.
    
    Compatible keys are:
    - Same key (e.g., 8A compatible with 8A)
    - ±1 on the wheel (e.g., 8A compatible with 7A, 9A)
    - Same number, different mode (e.g., 8A compatible with 8B)
    
    Args:
        camelot_key: String in Camelot notation (e.g., "8A")
    
    Returns:
        List of compatible Camelot keys
    """
    if not camelot_key or pd.isna(camelot_key):
        return []
    
    number = int(camelot_key[:-1])
    letter = camelot_key[-1]
    
    compatible = [camelot_key]  # Same key
    
    # ±1 on the wheel
    prev_number = number - 1 if number > 1 else 12
    next_number = number + 1 if number < 12 else 1
    compatible.append(f"{prev_number}{letter}")
    compatible.append(f"{next_number}{letter}")
    
    # Same number, different mode
    other_letter = "B" if letter == "A" else "A"
    compatible.append(f"{number}{other_letter}")
    
    return list(set(compatible))  # Remove duplicates


def similarity_ratio(str1, str2):
    """
    Calculate similarity ratio between two strings using SequenceMatcher.
    
    Args:
        str1: First string
        str2: Second string
    
    Returns:
        Float between 0 and 1 (1.0 = identical)
    """
    if pd.isna(str1) or pd.isna(str2):
        return 0.0
    return SequenceMatcher(None, str(str1).lower(), str(str2).lower()).ratio()


def deduplicate_search_results(results):
    """
    Remove duplicate songs from search results.
    Deduplicates by track_name + artists combination, keeping first occurrence.
    
    Args:
        results: DataFrame with search results
    
    Returns:
        DataFrame with duplicates removed
    """
    if len(results) == 0:
        return results
    
    # Create unique key from track_name and artists
    results = results.copy()
    results['_unique_key'] = (
        results['track_name'].astype(str).str.lower() + '|' + 
        results['artists'].astype(str).str.lower()
    )
    
    # Remove duplicates, keeping first occurrence
    results = results.drop_duplicates(subset=['_unique_key'], keep='first')
    results = results.drop('_unique_key', axis=1)
    
    return results


def search_by_track_id(dataset, track_id):
    """
    Search for a song by exact track ID match.
    
    Args:
        dataset: DataFrame with track data
        track_id: Spotify track ID string
    
    Returns:
        DataFrame row or None if not found
    """
    result = dataset[dataset['track_id'] == track_id]
    if len(result) > 0:
        return result.iloc[0]
    return None


def search_by_track_name(dataset, track_name, fuzzy=True):
    """
    Search for songs by track name. Exact matches preferred, then fuzzy matches.
    
    Args:
        dataset: DataFrame with track data
        track_name: Track name to search for
        fuzzy: Whether to use fuzzy matching if exact match not found
    
    Returns:
        DataFrame with matching tracks, sorted by match quality (exact first)
    """
    track_name_lower = str(track_name).lower()
    
    # Exact match (case-insensitive)
    exact_matches = dataset[
        dataset['track_name'].str.lower() == track_name_lower
    ]
    
    if len(exact_matches) > 0:
        return deduplicate_search_results(exact_matches)
    
    if not fuzzy:
        return pd.DataFrame()
    
    # Partial match (contains)
    partial_matches = dataset[
        dataset['track_name'].str.lower().str.contains(track_name_lower, na=False)
    ]
    
    if len(partial_matches) > 0:
        return deduplicate_search_results(partial_matches)
    
    # Fuzzy match
    dataset['_similarity'] = dataset['track_name'].apply(
        lambda x: similarity_ratio(x, track_name)
    )
    fuzzy_matches = dataset[dataset['_similarity'] > 0.6].sort_values(
        '_similarity', ascending=False
    )
    fuzzy_matches = fuzzy_matches.drop('_similarity', axis=1)
    
    return deduplicate_search_results(fuzzy_matches)


def search_by_artist(dataset, artist, fuzzy=True):
    """
    Search for songs by artist name. Handles multiple artists separated by ';'.
    
    Args:
        dataset: DataFrame with track data
        artist: Artist name to search for
        fuzzy: Whether to use fuzzy matching if exact match not found
    
    Returns:
        DataFrame with matching tracks, sorted by match quality
    """
    artist_lower = str(artist).lower()
    
    # Exact match (case-insensitive) - check if artist appears in artists column
    exact_matches = dataset[
        dataset['artists'].str.lower().str.contains(artist_lower, na=False, regex=False)
    ]
    
    if len(exact_matches) > 0:
        return deduplicate_search_results(exact_matches)
    
    if not fuzzy:
        return pd.DataFrame()
    
    # Fuzzy match on artists
    dataset['_similarity'] = dataset['artists'].apply(
        lambda x: similarity_ratio(x, artist)
    )
    fuzzy_matches = dataset[dataset['_similarity'] > 0.5].sort_values(
        '_similarity', ascending=False
    )
    fuzzy_matches = fuzzy_matches.drop('_similarity', axis=1)
    
    return deduplicate_search_results(fuzzy_matches)


def search_by_artist_and_track(dataset, artist, track_name, fuzzy=True):
    """
    Search for songs by both artist and track name. Exact matches preferred.
    
    Args:
        dataset: DataFrame with track data
        artist: Artist name
        track_name: Track name
        fuzzy: Whether to use fuzzy matching if exact match not found
    
    Returns:
        DataFrame with matching tracks, sorted by match quality
    """
    artist_lower = str(artist).lower()
    track_name_lower = str(track_name).lower()
    
    # Exact match on both
    exact_matches = dataset[
        (dataset['artists'].str.lower().str.contains(artist_lower, na=False, regex=False)) &
        (dataset['track_name'].str.lower() == track_name_lower)
    ]
    
    if len(exact_matches) > 0:
        return deduplicate_search_results(exact_matches)
    
    # Partial match on both
    partial_matches = dataset[
        (dataset['artists'].str.lower().str.contains(artist_lower, na=False, regex=False)) &
        (dataset['track_name'].str.lower().str.contains(track_name_lower, na=False))
    ]
    
    if len(partial_matches) > 0:
        return deduplicate_search_results(partial_matches)
    
    if not fuzzy:
        return pd.DataFrame()
    
    # Fuzzy match
    dataset['_artist_sim'] = dataset['artists'].apply(
        lambda x: similarity_ratio(x, artist)
    )
    dataset['_track_sim'] = dataset['track_name'].apply(
        lambda x: similarity_ratio(x, track_name)
    )
    dataset['_combined_sim'] = (dataset['_artist_sim'] + dataset['_track_sim']) / 2
    
    fuzzy_matches = dataset[dataset['_combined_sim'] > 0.6].sort_values(
        '_combined_sim', ascending=False
    )
    fuzzy_matches = fuzzy_matches.drop(['_artist_sim', '_track_sim', '_combined_sim'], axis=1)
    
    return deduplicate_search_results(fuzzy_matches)


def search_by_index(dataset, index):
    """
    Search for a song by dataset row index.
    
    Args:
        dataset: DataFrame with track data
        index: Integer row index
    
    Returns:
        DataFrame row or None if index out of range
    """
    if 0 <= index < len(dataset):
        return dataset.iloc[index]
    return None


def format_song_info(song):
    """
    Format song information for display.
    
    Args:
        song: DataFrame row (Series) with song data
    
    Returns:
        Formatted string
    """
    if song is None:
        return "Song not found"
    
    # Handle Series from iterrows()
    if isinstance(song, pd.Series):
        if song.empty:
            return "Song not found"
        artists = song.get('artists', 'Unknown')
        track_name = song.get('track_name', 'Unknown')
        bpm = song.get('tempo', 'N/A')
        camelot = song.get('camelot_key', 'N/A')
        energy = song.get('energy', 'N/A')
    else:
        # Handle dictionary or other types
        artists = song.get('artists', 'Unknown') if hasattr(song, 'get') else 'Unknown'
        track_name = song.get('track_name', 'Unknown') if hasattr(song, 'get') else 'Unknown'
        bpm = song.get('tempo', 'N/A') if hasattr(song, 'get') else 'N/A'
        camelot = song.get('camelot_key', 'N/A') if hasattr(song, 'get') else 'N/A'
        energy = song.get('energy', 'N/A') if hasattr(song, 'get') else 'N/A'
    
    # Handle NaN values
    if pd.isna(artists):
        artists = 'Unknown'
    if pd.isna(track_name):
        track_name = 'Unknown'
    if pd.isna(camelot):
        camelot = 'N/A'
    
    # Format BPM and energy (handle if they're strings like 'N/A' or NaN)
    try:
        if bpm != 'N/A' and pd.notna(bpm):
            bpm_str = f"{float(bpm):.1f}"
        else:
            bpm_str = "N/A"
    except (ValueError, TypeError):
        bpm_str = "N/A"
    
    try:
        if energy != 'N/A' and pd.notna(energy):
            energy_str = f"{float(energy):.2f}"
        else:
            energy_str = "N/A"
    except (ValueError, TypeError):
        energy_str = "N/A"
    
    return f"{track_name} by {artists} | BPM: {bpm_str} | Key: {camelot} | Energy: {energy_str}"
