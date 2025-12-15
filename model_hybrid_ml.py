"""
Model 3: Hybrid ML Recommendation System
Combines DJ mixing rules with learned audio patterns using XGBoost
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import (calculate_bpm_distance, is_key_compatible, 
                   calculate_energy_flow_score, get_camelot_notation)

class HybridMLRecommender:
    """
    Hybrid ML recommender combining rule-based features with learned patterns
    """
    
    def __init__(self, df):
        """
        Initialize recommender
        
        Args:
            df: DataFrame with song features
        """
        self.df = df.copy()
        self.model = None
        self.scaler = StandardScaler()
        
        print(f"Initialized Hybrid ML Recommender with {len(df)} songs")
    
    def create_training_pairs(self, n_pairs=10000, positive_ratio=0.5):
        """
        Create training pairs of (source_song, candidate_song, label)
        
        Args:
            n_pairs: Number of pairs to generate
            positive_ratio: Ratio of compatible pairs
        
        Returns:
            DataFrame: Training pairs with features and labels
        """
        print(f"\nGenerating {n_pairs} training pairs...")
        
        pairs = []
        n_positive = int(n_pairs * positive_ratio)
        n_negative = n_pairs - n_positive
        
        # Generate positive pairs (compatible songs)
        for _ in range(n_positive):
            # Random source song
            idx1 = np.random.randint(0, len(self.df))
            song1 = self.df.iloc[idx1]
            
            # Find compatible songs based on rules
            compatible = self.df[
                (abs(self.df['tempo'] - song1['tempo']) <= 6) &
                (self.df.apply(lambda row: is_key_compatible(
                    song1['key'], song1['mode'], row['key'], row['mode']), axis=1))
            ]
            
            if len(compatible) > 1:
                idx2 = np.random.choice(compatible.index)
                song2 = self.df.loc[idx2]
                pairs.append(self._create_pair_features(song1, song2, label=1))
        
        # Generate negative pairs (incompatible songs)
        for _ in range(n_negative):
            idx1 = np.random.randint(0, len(self.df))
            idx2 = np.random.randint(0, len(self.df))
            
            song1 = self.df.iloc[idx1]
            song2 = self.df.iloc[idx2]
            
            # Make sure they're actually incompatible
            bpm_dist, _ = calculate_bpm_distance(song1['tempo'], song2['tempo'])
            key_compat = is_key_compatible(song1['key'], song1['mode'], 
                                          song2['key'], song2['mode'])
            
            # Only add if significantly incompatible
            if bpm_dist > 10 or not key_compat:
                pairs.append(self._create_pair_features(song1, song2, label=0))
        
        pairs_df = pd.DataFrame(pairs)
        print(f"Generated {len(pairs_df)} pairs ({pairs_df['label'].sum()} positive)")
        
        return pairs_df
    
    def _create_pair_features(self, song1, song2, label):
        """
        Create feature vector for a song pair
        
        Args:
            song1, song2: Song DataFrames
            label: 1 if compatible, 0 otherwise
        
        Returns:
            dict: Feature dictionary
        """
        # BPM features
        bpm_dist, _ = calculate_bpm_distance(song1['tempo'], song2['tempo'])
        bpm_ratio = song2['tempo'] / song1['tempo'] if song1['tempo'] > 0 else 1
        
        # Key features
        key_compatible = 1 if is_key_compatible(
            song1['key'], song1['mode'], song2['key'], song2['mode']) else 0
        key_distance = abs(song1['key'] - song2['key'])
        mode_match = 1 if song1['mode'] == song2['mode'] else 0
        
        # Energy features
        energy_flow = calculate_energy_flow_score(song1['energy'], song2['energy'])
        energy_diff = abs(song1['energy'] - song2['energy'])
        
        # Audio feature differences
        valence_diff = abs(song1['valence'] - song2['valence'])
        danceability_diff = abs(song1['danceability'] - song2['danceability'])
        
        features = {
            # BPM features (weight: 40%)
            'bpm_distance': bpm_dist,
            'bpm_ratio': bpm_ratio,
            'bpm_within_6': 1 if bpm_dist <= 6 else 0,
            
            # Key features (weight: 30%)
            'key_compatible': key_compatible,
            'key_distance': key_distance,
            'mode_match': mode_match,
            
            # Energy features (weight: 30%)
            'energy_flow_score': energy_flow,
            'energy_diff': energy_diff,
            'valence_diff': valence_diff,
            'danceability_diff': danceability_diff,
            
            # Label
            'label': label
        }
        
        return features
    
    def train(self, n_pairs=10000, test_size=0.2):
        """
        Train the XGBoost model
        
        Args:
            n_pairs: Number of training pairs to generate
            test_size: Fraction for test set
        
        Returns:
            dict: Training metrics
        """
        print("\nTraining Hybrid ML Model...")
        print("="*60)
        
        # Generate training pairs
        pairs = self.create_training_pairs(n_pairs)
        
        # Prepare features and labels
        feature_cols = [col for col in pairs.columns if col != 'label']
        X = pairs[feature_cols]
        y = pairs['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train)} pairs")
        print(f"Test set: {len(X_test)} pairs")
        
        # Train XGBoost
        print("\nTraining XGBoost classifier...")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        
        print(f"\nTraining accuracy: {train_acc:.2%}")
        print(f"Test accuracy: {test_acc:.2%}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Most Important Features:")
        print(importance.head())
        
        metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'feature_importance': importance
        }
        
        return metrics
    
    def predict_compatibility(self, song1, song2):
        """
        Predict compatibility score between two songs
        
        Args:
            song1, song2: Song DataFrames
        
        Returns:
            float: Compatibility probability (0-1)
        """
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")
        
        # Create feature vector
        features = self._create_pair_features(song1, song2, label=0)
        del features['label']
        
        # Predict
        X = pd.DataFrame([features])
        probability = self.model.predict_proba(X)[0, 1]  # Probability of class 1
        
        return probability
    
    def recommend(self, song_idx, n_recommendations=10, verbose=True, 
                  use_rule_filter=True):
        """
        Recommend songs using hybrid approach
        
        Args:
            song_idx: Index of source song
            n_recommendations: Number of recommendations
            verbose: Print details
            use_rule_filter: Apply BPM filter first
        
        Returns:
            DataFrame: Recommended songs
        """
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")
        
        source_song = self.df.iloc[song_idx]
        
        if verbose:
            print("="*60)
            print("SOURCE SONG:")
            print("="*60)
            camelot = get_camelot_notation(source_song['key'], source_song['mode'])
            print(f"BPM: {source_song['tempo']:.1f} | Key: {camelot} | "
                  f"Energy: {source_song['energy']:.2f}")
            if 'name' in source_song:
                print(f"Title: {source_song['name']}")
            print()
        
        # Optional: Filter by BPM first (hybrid approach)
        if use_rule_filter:
            candidates = self.df[
                abs(self.df['tempo'] - source_song['tempo']) <= 10
            ].copy()
            if verbose:
                print(f"BPM Pre-filter: {len(candidates)} candidates")
        else:
            candidates = self.df.copy()
        
        # Score all candidates with ML model
        scores = []
        for idx, candidate in candidates.iterrows():
            if idx != song_idx:  # Skip source song
                score = self.predict_compatibility(source_song, candidate)
                scores.append((idx, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N
        top_indices = [idx for idx, _ in scores[:n_recommendations]]
        top_scores = [score for _, score in scores[:n_recommendations]]
        
        recommendations = self.df.loc[top_indices].copy()
        recommendations['ml_score'] = top_scores
        
        if verbose:
            print(f"\nTop {n_recommendations} Recommendations:")
            print("-"*60)
            for i, (idx, row) in enumerate(recommendations.iterrows(), 1):
                camelot = get_camelot_notation(row['key'], row['mode'])
                print(f"{i}. ML Score: {row['ml_score']:.3f} | "
                      f"BPM: {row['tempo']:.1f} | Key: {camelot}")
                if 'name' in row:
                    print(f"   {row['name']}")
            print()
        
        return recommendations


# Example usage
if __name__ == "__main__":
    print("Testing Hybrid ML Recommender")
    print("="*60)
    
    # Create sample dataset
    np.random.seed(42)
    n_songs = 500  # Larger dataset for training
    
    sample_data = pd.DataFrame({
        'tempo': np.random.uniform(80, 180, n_songs),
        'key': np.random.randint(0, 12, n_songs),
        'mode': np.random.randint(0, 2, n_songs),
        'energy': np.random.uniform(0, 1, n_songs),
        'valence': np.random.uniform(0, 1, n_songs),
        'danceability': np.random.uniform(0, 1, n_songs),
    })
    
    # Initialize and train
    recommender = HybridMLRecommender(sample_data)
    metrics = recommender.train(n_pairs=5000)
    
    # Test recommendation
    print("\n" + "="*60)
    print("Testing recommendation for song index 0:")
    recommendations = recommender.recommend(0, n_recommendations=5, verbose=True)
    
    print("="*60)
    print("Hybrid model successfully combines rules + ML!")
