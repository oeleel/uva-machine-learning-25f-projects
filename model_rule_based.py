"""
Model 1: Rule-Based DJ Recommendation System
Uses hard constraints: BPM ±6 and Camelot Wheel key compatibility
"""

import pandas as pd
import numpy as np
from utils import (calculate_bpm_distance, is_key_compatible, 
                   get_camelot_notation, calculate_energy_flow_score)

class RuleBasedDJRecommender:
    """
    Rule-based recommendation system following traditional DJ mixing rules
    """
    
    def __init__(self, df, bpm_tolerance=6):
        """
        Initialize recommender with song database
        
        Args:
            df: DataFrame with song features
            bpm_tolerance: Maximum BPM difference (default ±6)
        """
        self.df = df.copy()
        self.bpm_tolerance = bpm_tolerance
        
        print(f"Initialized Rule-Based Recommender with {len(df)} songs")
        print(f"BPM tolerance: ±{bpm_tolerance}")
    
    def filter_bpm_compatible(self, source_bpm):
        """
        Filter songs within BPM tolerance
        
        Args:
            source_bpm: BPM of source song
        
        Returns:
            DataFrame: Songs within BPM range
        """
        # Calculate BPM distance for all songs
        bpm_distances = self.df['tempo'].apply(
            lambda x: calculate_bpm_distance(source_bpm, x, self.bpm_tolerance)[0]
        )
        
        # Filter compatible songs
        compatible = self.df[bpm_distances <= self.bpm_tolerance].copy()
        compatible['bpm_distance'] = bpm_distances[bpm_distances <= self.bpm_tolerance]
        
        return compatible
    
    def filter_key_compatible(self, df_subset, source_key, source_mode):
        """
        Filter songs with compatible keys
        
        Args:
            df_subset: DataFrame to filter
            source_key: Musical key (0-11)
            source_mode: 0=minor, 1=major
        
        Returns:
            DataFrame: Key-compatible songs
        """
        # Check key compatibility for each song
        compatible_mask = df_subset.apply(
            lambda row: is_key_compatible(source_key, source_mode, row['key'], row['mode']),
            axis=1
        )
        
        return df_subset[compatible_mask]
    
    def rank_by_compatibility(self, candidates, source_energy):
        """
        Rank candidate songs by overall compatibility
        
        Args:
            candidates: DataFrame of candidate songs
            source_energy: Energy level of source song
        
        Returns:
            DataFrame: Ranked songs
        """
        if len(candidates) == 0:
            return candidates
        
        # Calculate energy flow scores
        candidates['energy_score'] = candidates['energy'].apply(
            lambda x: calculate_energy_flow_score(source_energy, x)
        )
        
        # Calculate overall score (weighted combination)
        # Lower BPM distance is better, higher energy score is better
        candidates['overall_score'] = (
            0.5 * (1 - candidates['bpm_distance'] / self.bpm_tolerance) +  # BPM similarity
            0.5 * candidates['energy_score']  # Energy flow
        )
        
        # Sort by overall score (descending)
        ranked = candidates.sort_values('overall_score', ascending=False)
        
        return ranked
    
    def recommend(self, song_idx, n_recommendations=10, verbose=True):
        """
        Recommend songs for mixing
        
        Args:
            song_idx: Index of source song in DataFrame
            n_recommendations: Number of recommendations to return
            verbose: Print details
        
        Returns:
            DataFrame: Top N recommended songs
        """
        # Get source song
        source_song = self.df.iloc[song_idx]
        source_bpm = source_song['tempo']
        source_key = source_song['key']
        source_mode = source_song['mode']
        source_energy = source_song['energy']
        
        if verbose:
            print("="*60)
            print("SOURCE SONG:")
            print("="*60)
            camelot = get_camelot_notation(source_key, source_mode)
            print(f"BPM: {source_bpm:.1f} | Key: {camelot} | Energy: {source_energy:.2f}")
            if 'name' in source_song:
                print(f"Title: {source_song['name']}")
            print()
        
        # Step 1: Filter by BPM
        bpm_compatible = self.filter_bpm_compatible(source_bpm)
        if verbose:
            print(f"Step 1 - BPM Filter (±{self.bpm_tolerance}): {len(bpm_compatible)} songs")
        
        # Step 2: Filter by key
        key_compatible = self.filter_key_compatible(bpm_compatible, source_key, source_mode)
        if verbose:
            print(f"Step 2 - Key Filter: {len(key_compatible)} songs")
        
        # Step 3: Rank by overall compatibility
        ranked = self.rank_by_compatibility(key_compatible, source_energy)
        
        # Remove source song itself
        ranked = ranked[ranked.index != song_idx]
        
        # Get top N
        recommendations = ranked.head(n_recommendations)
        
        if verbose:
            print(f"\nTop {n_recommendations} Recommendations:")
            print("-"*60)
            for i, (idx, row) in enumerate(recommendations.iterrows(), 1):
                camelot = get_camelot_notation(row['key'], row['mode'])
                print(f"{i}. BPM: {row['tempo']:.1f} | Key: {camelot} | "
                      f"Energy: {row['energy']:.2f} | Score: {row['overall_score']:.2f}")
                if 'name' in row:
                    print(f"   {row['name']}")
            print()
        
        return recommendations
    
    def batch_evaluate(self, test_indices, n_recommendations=10):
        """
        Evaluate on multiple test songs
        
        Args:
            test_indices: List of song indices to test
            n_recommendations: Number of recommendations per song
        
        Returns:
            dict: Evaluation metrics
        """
        results = {
            'bpm_compatible': [],
            'key_compatible': [],
            'avg_score': []
        }
        
        for idx in test_indices:
            recs = self.recommend(idx, n_recommendations, verbose=False)
            
            if len(recs) > 0:
                # Calculate metrics
                bpm_compat = (recs['bpm_distance'] <= self.bpm_tolerance).mean()
                results['bpm_compatible'].append(bpm_compat)
                
                # Key compatibility is implicit (we filtered for it)
                results['key_compatible'].append(1.0)
                
                results['avg_score'].append(recs['overall_score'].mean())
        
        # Aggregate results
        metrics = {
            'bpm_compatibility_rate': np.mean(results['bpm_compatible']),
            'key_compatibility_rate': np.mean(results['key_compatible']),
            'avg_overall_score': np.mean(results['avg_score']),
            'songs_tested': len(test_indices)
        }
        
        return metrics


# Example usage and testing
if __name__ == "__main__":
    print("Testing Rule-Based DJ Recommender")
    print("="*60)
    
    # Create sample dataset for testing
    np.random.seed(42)
    n_songs = 100
    
    sample_data = pd.DataFrame({
        'tempo': np.random.uniform(80, 180, n_songs),
        'key': np.random.randint(0, 12, n_songs),
        'mode': np.random.randint(0, 2, n_songs),
        'energy': np.random.uniform(0, 1, n_songs),
        'valence': np.random.uniform(0, 1, n_songs),
        'danceability': np.random.uniform(0, 1, n_songs),
    })
    
    # Initialize recommender
    recommender = RuleBasedDJRecommender(sample_data, bpm_tolerance=6)
    
    # Test recommendation for first song
    print("\nTesting recommendation for song index 0:")
    recommendations = recommender.recommend(0, n_recommendations=5, verbose=True)
    
    # Batch evaluation
    print("\nBatch Evaluation on 10 random songs:")
    test_indices = np.random.choice(len(sample_data), 10, replace=False)
    metrics = recommender.batch_evaluate(test_indices, n_recommendations=10)
    
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2%}" if 'rate' in key or 'score' in key else f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("To use with real data:")
    print("1. Load your processed Spotify dataset")
    print("2. Initialize: recommender = RuleBasedDJRecommender(df)")
    print("3. Get recommendations: recommender.recommend(song_idx)")
