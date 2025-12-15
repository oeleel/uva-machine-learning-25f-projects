# Quick Start Implementation Guide

## Step-by-Step Implementation for Your Team

### Phase 1: Setup (Week 1)

**Person 1 & 2: Data Setup**
1. Download Spotify dataset from Kaggle:
   - Link: https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset
   - Or search for any Spotify dataset with: tempo, key, mode, energy, valence, danceability
2. Create `data/` folder and place CSV inside
3. Run `data_preprocessing.py` to clean and prepare data
4. Verify Camelot notation is added correctly

**Person 3 & 4: Code Testing**
1. Install dependencies: `pip install -r requirements.txt`
2. Test `utils.py` functions work correctly
3. Run each model file individually to verify imports work

### Phase 2: Model Implementation (Week 2)

**Person 1: Rule-Based Model**
- File: `model_rule_based.py`
- Tasks:
  1. Test BPM filtering with different tolerances (±4, ±6, ±8)
  2. Verify key compatibility using Camelot Wheel
  3. Run batch evaluation on 50-100 songs
  4. Document results in notebook

**Person 2: Audio Similarity Baseline**
- File: `model_audio_similarity.py`
- Tasks:
  1. Experiment with different feature combinations
  2. Try different similarity metrics (cosine, euclidean)
  3. Evaluate how poorly it matches DJ rules (this is expected!)
  4. Document insights

**Person 3: Hybrid ML Model**
- File: `model_hybrid_ml.py`
- Tasks:
  1. Generate training pairs (start with 5000, scale to 10000+)
  2. Train XGBoost with different hyperparameters
  3. Analyze feature importance
  4. Compare performance vs baselines

**Person 4: Evaluation & Demo**
- Tasks:
  1. Create evaluation framework comparing all models
  2. Generate visualizations (BPM dist, key dist, model comparison)
  3. Build interactive demo in notebook
  4. Prepare demo video script

### Phase 3: Analysis (Week 3)

**Everyone: Run Full Pipeline**
1. Open `dj_mixing_analysis.ipynb` in Jupyter
2. Run each cell sequentially
3. Document results and insights

**Key Analyses to Complete:**
- BPM tolerance sensitivity (Person 1)
- Feature importance analysis (Person 3)
- Model comparison charts (Person 4)
- Edge case testing: half-tempo, key changes (Person 2)

### Phase 4: Final Deliverables (Week 4)

**Slide Deck (Everyone):**
- Use your Shark Tank presentation as template
- Add results slides with:
  - Model comparison table
  - Feature importance chart
  - Example recommendations
  - Key insights

**Demo Video (Person 4 lead, all participate):**
1. Script outline:
   - Intro: Problem statement (30 sec)
   - Show source song (10 sec)
   - Demo all 3 models side-by-side (60 sec)
   - Show evaluation metrics (30 sec)
   - Conclusion: Hybrid wins! (20 sec)
2. Record using Zoom/OBS
3. Upload to YouTube (unlisted)
4. Add link to slide deck

**GitHub PR:**
1. Create folder: `team-dj-mixing/`
2. Structure:
   ```
   team-dj-mixing/
   ├── src/
   │   ├── data_preprocessing.py
   │   ├── model_rule_based.py
   │   ├── model_audio_similarity.py
   │   ├── model_hybrid_ml.py
   │   └── utils.py
   ├── doc/
   │   ├── DJ_Mixing_Recommendation_Final.pptx
   │   └── dj_mixing_analysis.ipynb
   └── README.md
   ```
3. Submit PR to course repository
4. Submit to Canvas with YouTube link

## Expected Results

Based on your hypothesis, you should find:

**Rule-Based:**
- BPM Compatibility: ~100% (by design)
- Key Compatibility: ~100% (by design)
- Limitation: May have limited recommendations for some songs

**Audio Similarity:**
- BPM Compatibility: ~30-40% (poor!)
- Key Compatibility: ~20-30% (poor!)
- Insight: Pure audio similarity doesn't work for DJ mixing

**Hybrid ML:**
- BPM Compatibility: ~85-95% (best of both!)
- Key Compatibility: ~80-90% (learned patterns)
- Feature importance: BPM distance, key compatibility top features

## Troubleshooting

**Problem: Not enough compatible songs**
- Solution: Increase BPM tolerance to ±8 or ±10
- Or use larger dataset

**Problem: XGBoost training too slow**
- Solution: Reduce n_pairs to 5000
- Or reduce n_estimators to 50

**Problem: Missing columns in dataset**
- Solution: Check column names in your CSV
- Update feature lists in code to match your data

**Problem: Low accuracy in hybrid model**
- Solution: Generate more training pairs (15000+)
- Try different feature weights
- Experiment with XGBoost hyperparameters

## Timeline

- **Dec 14-15**: Data setup, code testing
- **Dec 15-16**: Run all three models
- **Dec 16**: Complete analysis in notebook
- **Dec 17 AM**: Record demo video
- **Dec 17 PM**: Submit everything by deadline

## Contact

If you get stuck, check:
1. README.md for detailed documentation
2. Code comments in each file
3. Course Piazza for dataset help
4. Office hours for ML model tuning

Good luck! You have a great project idea and solid implementation plan! 🎵🎧
