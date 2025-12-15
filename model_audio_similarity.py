"""
Model 2: Audio Feature Similarity Baseline
Content-based filtering using cosine similarity on audio features
Ignores BPM/key constraints to see if pure audio similarity works
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class AudioSimilarityRecommender:
    """
    Baseline recommender using audio feature similarity
    """
    
    def __init__(self, df, features=['energy', 'valence', 'danceability', 
                                     'acousticness', 'instrumentalness', 'loudness']):
        """
        Initialize recommender with audio features
        
        Args:
            df: DataFrame with song features
            features: List of feature columns to use
        """
        self.df = df.copy()
        self.features = [f for f in features if f in df.columns]
        
        if len(self.features) == 0:
            print("WARNING: No valid features found!")
            return
        
        print(f"Initialized Audio Similarity Recommender with {len(df)} songs")
        print(f"Using features: {self.features}")
        
        # Normalize features
        self.scaler = StandardScaler()
        self.feature_matrix = self.scaler.fit_transform(df[self.features])
        
        # Compute similarity matrix (can be memory intensive for large datasets)
        print("Computing similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        print("Similarity matrix computed!")
    
    def recommend(self, song_idx, n_recommendations=10, verbose=True):
        """
        Recommend songs based on audio similarity
        
        Args:
            song_idx: Index of source song
            n_recommendations: Number of recommendations
            verbose: Print details
        
        Returns:
            DataFrame: Top N similar songs
        """
        # Get source song
        source_song = self.df.iloc[song_idx]
        
        if verbose:
            print("="*60)
            print("SOURCE SONG:")
            print("="*60)
            if 'name' in source_song:
                print(f"Title: {source_song['name']}")
            print(f"Audio features:")
            for feat in self.features:
                print(f"  {feat}: {source_song[feat]:.2f}")
            print()
        
        # Get similarity scores for all songs
        similarities = self.similarity_matrix[song_idx]
        
        # Create results DataFrame
        results = self.df.copy()
        results['similarity_score'] = similarities
        
        # Sort by similarity (descending)
        results = results.sort_values('similarity_score', ascending=False)
        
        # Remove source song itself
        results = results[results.index != song_idx]
        
        # Get top N
        recommendations = results.head(n_recommendations)
        
        if verbose:
            print(f"Top {n_recommendations} Recommendations:")
            print("-"*60)
            for i, (idx, row) in enumerate(recommendations.iterrows(), 1):
                print(f"{i}. Similarity: {row['similarity_score']:.3f}")
                if 'name' in row:
                    print(f"   {row['name']}")
                # Show audio features
                print(f"   ", end="")
                for feat in self.features[:3]:  # Show first 3 features
                    print(f"{feat}: {row[feat]:.2f} | ", end="")
                print()
            print()
        
        return recommendations
    
    def evaluate_mixing_quality(self, recommendations, source_idx):
        """
        Evaluate how well recommendations match DJ mixing rules
        (even though this model doesn't use them)
        
        Args:
            recommendations: DataFrame of recommended songs
            source_idx: Index of source song
        
        Returns:
            dict: Mixing quality metrics
        """
        from utils import calculate_bpm_distance, is_key_compatible
        
        source_song = self.df.iloc[source_idx]
        source_bpm = source_song['tempo']
        source_key = source_song['key']
        source_mode = source_song['mode']
        
        # Check BPM compatibility (±6 BPM rule)
        bpm_compatible = 0
        for idx, row in recommendations.iterrows():
            _, is_compat = calculate_bpm_distance(source_bpm, row['tempo'], tolerance=6)
            if is_compat:
                bpm_compatible += 1
        
        # Check key compatibility
        key_compatible = 0
        for idx, row in recommendations.iterrows():
            if is_key_compatible(source_key, source_mode, row['key'], row['mode']):
                key_compatible += 1
        
        n_recs = len(recommendations)
        metrics = {
            'bpm_compatibility_rate': bpm_compatible / n_recs if n_recs > 0 else 0,
            'key_compatibility_rate': key_compatible / n_recs if n_recs > 0 else 0,
            'avg_similarity': recommendations['similarity_score'].mean() if n_recs > 0 else 0
        }
        
        return metrics
    
    def batch_evaluate(self, test_indices, n_recommendations=10):
        """
        Evaluate on multiple test songs
        
        Args:
            test_indices: List of song indices
            n_recommendations: Number of recommendations per song
        
        Returns:
            dict: Aggregated metrics
        """
        all_metrics = {
            'bpm_compatible': [],
            'key_compatible': [],
            'avg_similarity': []
        }
        
        for idx in test_indices:
            recs = self.recommend(idx, n_recommendations, verbose=False)
            metrics = self.evaluate_mixing_quality(recs, idx)
            
            all_metrics['bpm_compatible'].append(metrics['bpm_compatibility_rate'])
            all_metrics['key_compatible'].append(metrics['key_compatibility_rate'])
            all_metrics['avg_similarity'].append(metrics['avg_similarity'])
        
        # Aggregate
        final_metrics = {
            'bpm_compatibility_rate': np.mean(all_metrics['bpm_compatible']),
            'key_compatibility_rate': np.mean(all_metrics['key_compatible']),
            'avg_similarity_score': np.mean(all_metrics['avg_similarity']),
            'songs_tested': len(test_indices)
        }
        
        return final_metrics
    
    def get_feature_importance(self, song_idx, recommendation_idx):
        """
        Analyze which features contribute most to similarity
        
        Args:
            song_idx: Source song index
            recommendation_idx: Recommended song index
        
        Returns:
            dict: Feature contributions
        """
        source_features = self.feature_matrix[song_idx]
        rec_features = self.feature_matrix[recommendation_idx]
        
        # Calculate contribution of each feature
        contributions = {}
        for i, feat in enumerate(self.features):
            diff = abs(source_features[i] - rec_features[i])
            contributions[feat] = 1 - diff  # Higher = more similar
        
        return contributions


# Example usage and testing
if __name__ == "__main__":
    print("Testing Audio Similarity Recommender")
    print("="*60)
    
    # Create sample dataset
    np.random.seed(42)
    n_songs = 100
    
    sample_data = pd.DataFrame({
        'tempo': np.random.uniform(80, 180, n_songs),
        'key': np.random.randint(0, 12, n_songs),
        'mode': np.random.randint(0, 2, n_songs),
        'energy': np.random.uniform(0, 1, n_songs),
        'valence': np.random.uniform(0, 1, n_songs),
        'danceability': np.random.uniform(0, 1, n_songs),
        'acousticness': np.random.uniform(0, 1, n_songs),
        'instrumentalness': np.random.uniform(0, 1, n_songs),
        'loudness': np.random.uniform(-60, 0, n_songs),
    })
    
    # Initialize recommender
    recommender = AudioSimilarityRecommender(sample_data)
    
    # Test recommendation
    print("\nTesting recommendation for song index 0:")
    recommendations = recommender.recommend(0, n_recommendations=5, verbose=True)
    
    # Evaluate mixing quality
    print("Evaluating DJ mixing quality:")
    metrics = recommender.evaluate_mixing_quality(recommendations, 0)
    print("\nMixing Quality Metrics:")
    for key, value in metrics.items():
        if 'rate' in key:
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value:.3f}")
    
    # Batch evaluation
    print("\n" + "="*60)
    print("Batch Evaluation on 10 random songs:")
    test_indices = np.random.choice(len(sample_data), 10, replace=False)
    batch_metrics = recommender.batch_evaluate(test_indices, n_recommendations=10)
    
    print("\nAggregated Metrics:")
    for key, value in batch_metrics.items():
        if 'rate' in key or 'score' in key:
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("Key Insight:")
    print("Audio similarity alone achieves ~{:.1%} BPM compatibility".format(
        batch_metrics['bpm_compatibility_rate']))
    print("This shows why DJ-aware systems are needed!")
