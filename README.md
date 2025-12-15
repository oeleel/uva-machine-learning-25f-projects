<<<<<<< HEAD
# DJ Mixing Recommendation System

A machine learning system that recommends mixable songs for DJs based on BPM, musical key, and energy flow. The system implements three models: Rule-Based, Audio Similarity Baseline, and Hybrid ML (XGBoost).

## Project Structure

```
team-28/
├── src/                    # All source code
│   ├── main.py            # Main execution script
│   ├── data_preprocessing.py
│   ├── model_rule_based.py
│   ├── model_audio_similarity.py
│   ├── model_hybrid_ml.py
│   ├── evaluation.py
│   ├── utils.py
│   └── visualize_results.py
├── doc/                    # Documentation
│   └── QUICKSTART.md
├── data/                   # Dataset
│   └── dataset.csv
├── requirements.txt        # Python dependencies
├── hybrid_model.pkl        # Trained ML model (generated)
└── README.md              # This file
```

## Features

- **Three Recommendation Models:**
  - Rule-Based: Uses hard constraints (BPM ±6, key compatibility)
  - Audio Similarity: Content-based filtering using cosine similarity
  - Hybrid ML: XGBoost model combining DJ rules with audio features

- **Multiple Search Methods:**
  - Search by track ID, song name, artist, or dataset index
  - Fuzzy matching with exact match preference

- **DJ Mixing Rules:**
  - BPM compatibility (±6 BPM tolerance)
  - Camelot Wheel key compatibility
  - Energy flow analysis
  - Genre matching

## Usage

### Basic Usage

Run from the project root directory:

```bash
# Search by song and artist
python src/main.py --song "Strobe" --artist "deadmau5"

# Search by track ID
python src/main.py --track_id 5SuOikwiRyPMVoIQDJUgSV

# Search by song name only
python src/main.py --song "Strobe"

# Search by artist
python src/main.py --artist "deadmau5"

# Search by index
python src/main.py --index 0
```

### Command-Line Options

```
--track_id <id>        Spotify track ID (exact match)
--song <name>          Track name (fuzzy matching)
--artist <name>        Artist name (fuzzy matching)
--index <number>       Dataset row index
--data <path>          Path to dataset CSV (default: ../data/dataset.csv)
--model_path <path>    Path to hybrid ML model (default: ../hybrid_model.pkl)
--train_model          Force retrain the hybrid ML model
--no_eval              Skip evaluation metrics
```

### Generate Visualizations

```bash
python src/visualize_results.py
```

## Documentation

See `doc/QUICKSTART.md` for detailed usage instructions and troubleshooting.

## Requirements

- Python 3.7+
- pandas >= 1.5.0
- numpy >= 1.23.0
- xgboost >= 1.7.0
- scikit-learn >= 1.1.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0

=======
DJ Song Mixing Recommendation System
Team: Ashley Wu, Bonny Koo, Nathan Suh, Leo Lee
Course: CS 4774 Machine Learning - UVA Fall 2025
Overview
A machine learning system that recommends songs for seamless DJ transitions by combining music theory (BPM, harmonic key compatibility) with audio features. The system compares three approaches: rule-based DJ constraints, audio similarity, and a hybrid ML model.
Problem Statement
DJs spend hours finding compatible songs for mixing. This system automates the process by recommending songs that match based on:

BPM compatibility (±6 BPM ideal for beatmatching)
Key compatibility (Camelot Wheel harmonic mixing)
Energy flow for smooth transitions

Three Models
Model 1: Rule-Based System
Traditional DJ approach using hard constraints.

Filters songs within ±6 BPM
Requires Camelot Wheel key compatibility
Ranks by weighted score (40% BPM, 35% Key, 20% Energy, 5% Genre)

Model 2: Audio Similarity Baseline
Content-based filtering using cosine similarity.

Compares energy, valence, danceability, acousticness
Ignores BPM/key constraints
Finds similar-sounding songs (but often unmixable)

Model 3: Hybrid ML System
XGBoost classifier combining rules with learned patterns.

Trained on 10,000 song pairs labeled by DJ rules
Features: BPM distance, key compatibility, energy difference
Achieves 100% train/test accuracy

# Install dependencies
pip install -r requirements.txt
Requirements
pandas>=1.5.0
numpy>=1.23.0
xgboost>=1.7.0
scikit-learn>=1.1.0
Dataset
Download the Spotify dataset from Kaggle and place it at data/dataset.csv.
Usage
Command Line Interface
bash# Search by song name and artist
python main.py --song "Strobe" --artist "deadmau5"

# Search by song name only (fuzzy matching)
python main.py --song "Blinding Lights"

# Search by dataset index
python main.py --index 0

# Search by Spotify track ID
python main.py --track_id 5SuOikwiRyPMVoIQDJUgSV
Options
--track_id <id>      Spotify track ID (exact match)
--song <name>        Track name (fuzzy matching)
--artist <name>      Artist name (fuzzy matching)
--index <number>     Dataset row index
--data <path>        Path to dataset CSV (default: data/dataset.csv)
--model_path <path>  Path to hybrid ML model (default: hybrid_model.pkl)
--train_model        Force retrain the hybrid ML model
--no_eval            Skip evaluation metrics
Example Output
License
This project is for educational purposes as part of UVA CS 4774.
>>>>>>> 2adf59f06f06e40a623ad4a9818e986617149667
