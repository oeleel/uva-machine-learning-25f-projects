# DJ Song Mixing Recommendation System

**Team:** Ashley Wu, Bonny Koo, Nathan Suh, Leo Lee  
**Course:** CS Machine Learning - UVA Fall 2025

## Project Overview

A machine learning system that recommends songs for seamless DJ transitions by combining music theory (BPM, harmonic key compatibility) with audio features.

### Problem Statement
DJs spend hours finding compatible songs for mixing. This system automates the process by recommending songs that match based on:
- BPM compatibility (±6 BPM ideal)
- Key compatibility (Camelot Wheel harmonic mixing)
- Energy flow for smooth transitions

## Project Structure

```
team-dj-mixing/
├── src/
│   ├── data_preprocessing.py      # Load and clean Spotify dataset
│   ├── feature_engineering.py     # BPM distance, key compatibility features
│   ├── model_rule_based.py        # Rule-based DJ system (Model 1)
│   ├── model_audio_similarity.py  # Audio similarity baseline (Model 2)
│   ├── model_hybrid_ml.py         # Hybrid ML system (Model 3)
│   ├── evaluation.py              # Compare all models
│   ├── utils.py                   # Camelot wheel, helper functions
│   └── demo.py                    # Interactive demo
├── doc/
│   ├── project_slides.pptx        # Final presentation
│   └── analysis_notebook.ipynb    # Main Jupyter notebook
├── data/
│   └── spotify_data.csv           # Dataset (download separately)
└── README.md
```

## Three Models to Implement

### Model 1: Rule-Based DJ System
- **Approach:** Hard constraints on BPM (±6) and key compatibility
- **Pros:** Follows DJ theory precisely
- **Cons:** Rigid, may miss creative combinations

### Model 2: Audio Feature Similarity (Baseline)
- **Approach:** Content-based filtering using energy, valence, danceability
- **Pros:** Discovers unexpected matches
- **Cons:** May recommend unmixable songs (BPM/key incompatible)

### Model 3: Hybrid ML System (Primary)
- **Approach:** XGBoost/LightGBM trained on weighted features
  - 40% BPM compatibility
  - 30% Key compatibility
  - 30% Energy/audio features
- **Pros:** Balances rules + discovery
- **Cons:** Needs training data

## Implementation Steps

### Step 1: Data Collection & Preprocessing
1. Download Spotify dataset from Kaggle
2. Extract key features: BPM, key, mode, energy, valence, danceability
3. Clean missing values
4. Convert musical keys to Camelot notation

### Step 2: Feature Engineering
1. Calculate BPM distance between songs
2. Create key compatibility matrix (Camelot Wheel)
3. Compute energy flow scores
4. Generate training labels (compatible vs incompatible pairs)

### Step 3: Build Rule-Based System
1. Implement BPM filtering (±6 BPM)
2. Implement key compatibility checker
3. Rank by weighted distance metric

### Step 4: Build Audio Similarity Baseline
1. Use cosine similarity on audio features
2. Ignore BPM/key constraints
3. Rank by feature similarity

### Step 5: Build Hybrid ML Model
1. Create training data from rule-based labels
2. Train XGBoost classifier
3. Combine predictions with rule-based filtering

### Step 6: Evaluation
1. Test all 3 models on held-out songs
2. Metrics: BPM accuracy, Key compatibility %, Energy smoothness
3. DJ validation (qualitative assessment)

### Step 7: Demo & Visualization
1. Interactive song selector
2. Show top 10 recommendations
3. Visualize BPM/key/energy comparisons

## Datasets

**Primary:** Spotify Dataset (Kaggle)
- Contains: BPM, key, mode, energy, valence, danceability, acousticness
- Size: 100k+ tracks
- Link: https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset

## Tools & Libraries

```python
# Data processing
import pandas as pd
import numpy as np

# Machine learning
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Demo
import gradio as gr  # Optional: for web interface
```

## Expected Analysis

### Research Questions
1. Which feature matters most? (BPM vs Key vs Energy)
2. Optimal BPM tolerance? (±4 vs ±6 vs ±8)
3. Does genre affect mixing rules?
4. Can ML discover non-obvious mixing patterns?

### Edge Cases
- Half-tempo transitions (128 BPM → 64 BPM)
- Key changes during breakdowns
- Vocal vs instrumental tracks
- Build-up to drop transitions

## Evaluation Metrics

1. **BPM Compatibility Rate:** % recommendations within ±6 BPM
2. **Key Compatibility Rate:** % recommendations with compatible keys
3. **Energy Flow Score:** Smoothness of energy transitions
4. **DJ Validation:** Qualitative feedback from experienced DJs
5. **Model Comparison:** Rule-based vs Audio vs Hybrid performance

## Deliverables (Due Dec 17)

- [x] Shark Tank pitch (completed)
- [ ] Slide deck with results
- [ ] Jupyter notebook with all models
- [ ] Demo video (YouTube link)
- [ ] GitHub PR to course repository

## Team Division

Suggested task split for 4 people:
- **Person 1:** Data preprocessing + Rule-based model
- **Person 2:** Audio similarity baseline + Feature engineering
- **Person 3:** Hybrid ML model + XGBoost training
- **Person 4:** Evaluation + Demo + Visualization

## References

- Camelot Wheel harmonic mixing system
- Mixed In Key software documentation
- Spotify API audio features documentation
- librosa for audio analysis
