"""
Data preprocessing for DJ mixing recommendation system
Loads Spotify dataset and prepares features
"""

import pandas as pd
import numpy as np
from utils import get_camelot_notation

def load_spotify_data(filepath='data/spotify_data.csv'):
    """
    Load Spotify dataset from CSV
    
    Expected columns:
    - tempo (BPM)
    - key (0-11)
    - mode (0=minor, 1=major)
    - energy (0-1)
    - valence (0-1)
    - danceability (0-1)
    - acousticness (0-1)
    - instrumentalness (0-1)
    - loudness (dB)
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        DataFrame: Cleaned dataset
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def clean_data(df):
    """
    Clean and preprocess the dataset
    
    Args:
        df: Raw DataFrame
    
    Returns:
        DataFrame: Cleaned DataFrame
    """
    print("\nCleaning data...")
    
    # Required columns for DJ mixing
    required_cols = ['tempo', 'key', 'mode', 'energy', 'valence', 'danceability']
    
    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols}")
        print("Available columns:", list(df.columns))
        return None
    
    # Remove rows with missing values in key columns
    df_clean = df.dropna(subset=required_cols)
    
    # Filter valid BPM range (typical DJ range: 80-180 BPM)
    df_clean = df_clean[(df_clean['tempo'] >= 80) & (df_clean['tempo'] <= 180)]
    
    # Filter valid key values (0-11)
    df_clean = df_clean[(df_clean['key'] >= 0) & (df_clean['key'] <= 11)]
    
    # Filter valid mode values (0 or 1)
    df_clean = df_clean[df_clean['mode'].isin([0, 1])]
    
    # Filter valid energy values (0-1)
    df_clean = df_clean[(df_clean['energy'] >= 0) & (df_clean['energy'] <= 1)]
    
    print(f"Cleaned dataset shape: {df_clean.shape}")
    print(f"Removed {len(df) - len(df_clean)} rows")
    
    return df_clean

def add_camelot_notation(df):
    """
    Add Camelot notation column to DataFrame
    
    Args:
        df: DataFrame with 'key' and 'mode' columns
    
    Returns:
        DataFrame: With 'camelot' column added
    """
    print("\nAdding Camelot notation...")
    df['camelot'] = df.apply(lambda row: get_camelot_notation(row['key'], row['mode']), axis=1)
    
    # Count distribution of keys
    print("\nKey distribution (Camelot notation):")
    print(df['camelot'].value_counts().head(10))
    
    return df

def create_genre_features(df):
    """
    Create genre-based features if genre column exists
    
    Args:
        df: DataFrame
    
    Returns:
        DataFrame: With genre features added
    """
    if 'track_genre' in df.columns or 'genre' in df.columns:
        genre_col = 'track_genre' if 'track_genre' in df.columns else 'genre'
        print(f"\nGenre distribution:")
        print(df[genre_col].value_counts().head(10))
    else:
        print("\nNo genre column found, skipping genre features")
    
    return df

def get_dataset_statistics(df):
    """
    Print dataset statistics
    
    Args:
        df: DataFrame
    """
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    print(f"\nTotal songs: {len(df)}")
    
    print("\nBPM Statistics:")
    print(f"  Mean: {df['tempo'].mean():.1f} BPM")
    print(f"  Median: {df['tempo'].median():.1f} BPM")
    print(f"  Range: {df['tempo'].min():.1f} - {df['tempo'].max():.1f} BPM")
    
    print("\nEnergy Statistics:")
    print(f"  Mean: {df['energy'].mean():.2f}")
    print(f"  Median: {df['energy'].median():.2f}")
    print(f"  Range: {df['energy'].min():.2f} - {df['energy'].max():.2f}")
    
    print("\nKey Distribution:")
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for i in range(12):
        count = len(df[df['key'] == i])
        print(f"  {key_names[i]}: {count} songs")
    
    print("\nMode Distribution:")
    print(f"  Minor (0): {len(df[df['mode'] == 0])} songs")
    print(f"  Major (1): {len(df[df['mode'] == 1])} songs")
    
    print("\n" + "="*60)

def prepare_dataset(filepath='data/spotify_data.csv'):
    """
    Complete preprocessing pipeline
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        DataFrame: Fully preprocessed dataset ready for modeling
    """
    # Load data
    df = load_spotify_data(filepath)
    
    # Clean data
    df = clean_data(df)
    
    if df is None:
        print("ERROR: Could not clean data. Check column names.")
        return None
    
    # Add Camelot notation
    df = add_camelot_notation(df)
    
    # Add genre features if available
    df = create_genre_features(df)
    
    # Print statistics
    get_dataset_statistics(df)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print("\nPreprocessing complete!")
    return df

def save_processed_data(df, output_path='data/spotify_processed.csv'):
    """
    Save processed dataset
    
    Args:
        df: Processed DataFrame
        output_path: Where to save
    """
    df.to_csv(output_path, index=False)
    print(f"\nSaved processed data to {output_path}")

# Example usage
if __name__ == "__main__":
    # This will be run when you execute this file directly
    print("DJ Mixing Recommendation - Data Preprocessing")
    print("="*60)
    
    # Example: Process the data
    # Uncomment when you have the dataset downloaded
    
    """
    df = prepare_dataset('data/spotify_data.csv')
    
    if df is not None:
        # Save processed data
        save_processed_data(df, 'data/spotify_processed.csv')
        
        # Show first few rows
        print("\nFirst 5 songs:")
        print(df[['name', 'artists', 'tempo', 'camelot', 'energy']].head() 
              if 'name' in df.columns else df[['tempo', 'camelot', 'energy']].head())
    """
    
    print("\nTo use this script:")
    print("1. Download Spotify dataset from Kaggle")
    print("2. Place it in data/spotify_data.csv")
    print("3. Uncomment the code above and run this file")
