"""
Utility functions for DJ mixing recommendation system
Includes Camelot Wheel implementation and helper functions
"""

import numpy as np
import pandas as pd

# Camelot Wheel mapping
# Maps Spotify key/mode to Camelot notation (used by DJs for harmonic mixing)
CAMELOT_WHEEL = {
    # Major keys (B notation)
    (0, 1): '8B',   # C Major
    (1, 1): '3B',   # C#/Db Major
    (2, 1): '10B',  # D Major
    (3, 1): '5B',   # D#/Eb Major
    (4, 1): '12B',  # E Major
    (5, 1): '7B',   # F Major
    (6, 1): '2B',   # F#/Gb Major
    (7, 1): '9B',   # G Major
    (8, 1): '4B',   # G#/Ab Major
    (9, 1): '11B',  # A Major
    (10, 1): '6B',  # A#/Bb Major
    (11, 1): '1B',  # B Major
    
    # Minor keys (A notation)
    (0, 0): '5A',   # C Minor
    (1, 0): '12A',  # C#/Db Minor
    (2, 0): '7A',   # D Minor
    (3, 0): '2A',   # D#/Eb Minor
    (4, 0): '9A',   # E Minor
    (5, 0): '4A',   # F Minor
    (6, 0): '11A',  # F#/Gb Minor
    (7, 0): '6A',   # G Minor
    (8, 0): '1A',   # G#/Ab Minor
    (9, 0): '8A',   # A Minor
    (10, 0): '3A',  # A#/Bb Minor
    (11, 0): '10A', # B Minor
}

# Reverse mapping for quick lookup
CAMELOT_TO_KEY = {v: k for k, v in CAMELOT_WHEEL.items()}

def get_camelot_notation(key, mode):
    """
    Convert Spotify key and mode to Camelot notation
    
    Args:
        key (int): Spotify key (0=C, 1=C#, ..., 11=B)
        mode (int): 0=minor, 1=major
    
    Returns:
        str: Camelot notation (e.g., '8A', '5B')
    """
    return CAMELOT_WHEEL.get((key, mode), 'Unknown')

def get_compatible_keys(camelot_key):
    """
    Get harmonically compatible keys according to Camelot Wheel
    
    Rules:
    - Same key (perfect match)
    - +1 or -1 on the wheel (energy shift)
    - Switch between A/B (relative major/minor)
    
    Args:
        camelot_key (str): e.g., '8A'
    
    Returns:
        list: Compatible Camelot keys
    """
    if camelot_key == 'Unknown':
        return []
    
    # Extract number and letter
    num = int(camelot_key[:-1])
    letter = camelot_key[-1]
    
    compatible = [camelot_key]  # Perfect match
    
    # +1 and -1 on wheel (energy shift)
    next_num = (num % 12) + 1
    prev_num = ((num - 2) % 12) + 1
    compatible.append(f"{next_num}{letter}")
    compatible.append(f"{prev_num}{letter}")
    
    # Relative major/minor (switch A/B)
    opposite_letter = 'B' if letter == 'A' else 'A'
    compatible.append(f"{num}{opposite_letter}")
    
    return compatible

def is_key_compatible(key1, mode1, key2, mode2):
    """
    Check if two songs are key compatible for mixing
    
    Args:
        key1, mode1: First song's key and mode
        key2, mode2: Second song's key and mode
    
    Returns:
        bool: True if compatible
    """
    camelot1 = get_camelot_notation(key1, mode1)
    camelot2 = get_camelot_notation(key2, mode2)
    
    compatible_keys = get_compatible_keys(camelot1)
    return camelot2 in compatible_keys

def calculate_bpm_distance(bpm1, bpm2, tolerance=6):
    """
    Calculate BPM distance and check compatibility
    
    Args:
        bpm1, bpm2: Tempos to compare
        tolerance: Maximum BPM difference (default ±6)
    
    Returns:
        tuple: (distance, is_compatible)
    """
    distance = abs(bpm1 - bpm2)
    
    # Check half-tempo compatibility (e.g., 128 and 64 BPM)
    half_tempo_distance = min(abs(bpm1 - 2*bpm2), abs(2*bpm1 - bpm2))
    
    # Use minimum distance
    min_distance = min(distance, half_tempo_distance)
    
    is_compatible = min_distance <= tolerance
    return min_distance, is_compatible

def calculate_energy_flow_score(energy1, energy2):
    """
    Calculate energy flow score (prefer smooth transitions)
    
    Args:
        energy1, energy2: Energy values (0-1)
    
    Returns:
        float: Score (0-1, higher is better)
    """
    # Penalize large jumps in energy
    energy_diff = abs(energy1 - energy2)
    
    # Smooth transition: difference < 0.2 is ideal
    # Large jump: difference > 0.5 is poor
    if energy_diff < 0.2:
        return 1.0
    elif energy_diff < 0.3:
        return 0.8
    elif energy_diff < 0.5:
        return 0.5
    else:
        return 0.2

def normalize_features(df, features):
    """
    Normalize feature columns using StandardScaler
    
    Args:
        df: DataFrame
        features: List of column names to normalize
    
    Returns:
        DataFrame: Normalized features
    """
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    df_norm = df.copy()
    df_norm[features] = scaler.fit_transform(df[features])
    return df_norm, scaler

def print_song_info(row):
    """
    Pretty print song information
    
    Args:
        row: DataFrame row with song info
    """
    camelot = get_camelot_notation(row['key'], row['mode'])
    print(f"🎵 {row.get('name', 'Unknown')} - {row.get('artists', 'Unknown')}")
    print(f"   BPM: {row['tempo']:.1f} | Key: {camelot} | Energy: {row['energy']:.2f}")
    print()

def calculate_overall_compatibility(bpm1, key1, mode1, energy1,
                                   bpm2, key2, mode2, energy2,
                                   bpm_weight=0.4, key_weight=0.3, energy_weight=0.3):
    """
    Calculate overall compatibility score between two songs
    
    Args:
        bpm1, key1, mode1, energy1: First song features
        bpm2, key2, mode2, energy2: Second song features
        bpm_weight, key_weight, energy_weight: Feature weights
    
    Returns:
        float: Overall compatibility score (0-1)
    """
    # BPM compatibility (inverse of normalized distance)
    bpm_dist, bpm_compatible = calculate_bpm_distance(bpm1, bpm2)
    bpm_score = max(0, 1 - bpm_dist / 20)  # Normalize to 0-1
    
    # Key compatibility (binary)
    key_score = 1.0 if is_key_compatible(key1, mode1, key2, mode2) else 0.0
    
    # Energy flow score
    energy_score = calculate_energy_flow_score(energy1, energy2)
    
    # Weighted combination
    overall_score = (bpm_weight * bpm_score + 
                    key_weight * key_score + 
                    energy_weight * energy_score)
    
    return overall_score

# Test the functions
if __name__ == "__main__":
    print("Testing Camelot Wheel functions...")
    print()
    
    # Test key conversion
    print("C Major (key=0, mode=1):", get_camelot_notation(0, 1))
    print("A Minor (key=9, mode=0):", get_camelot_notation(9, 0))
    print()
    
    # Test compatible keys
    print("Compatible keys with 8A (A minor):")
    print(get_compatible_keys('8A'))
    print()
    
    # Test BPM compatibility
    print("BPM compatibility tests:")
    print("128 vs 124 BPM:", calculate_bpm_distance(128, 124))
    print("128 vs 140 BPM:", calculate_bpm_distance(128, 140))
    print("128 vs 64 BPM (half-tempo):", calculate_bpm_distance(128, 64))
    print()
    
    # Test overall compatibility
    print("Overall compatibility:")
    score = calculate_overall_compatibility(
        bpm1=128, key1=9, mode1=0, energy1=0.8,  # Song 1: 128 BPM, A minor, high energy
        bpm2=124, key2=9, mode2=0, energy2=0.75  # Song 2: 124 BPM, A minor, high energy
    )
    print(f"Compatible songs score: {score:.2f}")
    
    score = calculate_overall_compatibility(
        bpm1=128, key1=9, mode1=0, energy1=0.8,  # Song 1: 128 BPM, A minor, high energy
        bpm2=140, key2=2, mode2=1, energy2=0.3   # Song 2: 140 BPM, D major, low energy
    )
    print(f"Incompatible songs score: {score:.2f}")
