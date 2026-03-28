# Smartphone Addiction Predictor 📱

A machine learning model that predicts smartphone addiction based on user behavior patterns. Trained on data from 7,500 users, this predictor analyzes key indicators to classify users as either "Addicted" or "Not Addicted".

**Status:** ✅ Production Ready | **Accuracy:** 86%+ | **Model:** Random Forest

## How It Works

```
User Inputs:
  • Screen time (hours/day)
  • Sleep hours
  • Gaming hours
  • Notifications (per day)
           ↓
    Model (trained on 7500 people)
           ↓
    Prediction Output:
      👉 "Addicted" or "Not Addicted"
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (first time only)
python src/train.py

# 3. Make predictions
python src/predict.py
```

**That's it!** See [Installation](#installation) below for detailed setup steps.

## Project Structure

```
smartphone-addiction-predictor/
├── src/
│   ├── train.py                 # Model training
│   └── predict.py               # Interactive predictions
├── data/
│   └── data.csv                 # Dataset with 7500+ records
├── models/                       # (Generated after training)
│   ├── addiction_model.pkl      # Trained model
│   ├── scaler.pkl               # Feature scaler
│   └── features.pkl             # Feature list
├── notebook/
│   └── analysis.ipynb           # Data exploration
├── docs/
│   ├── PROJECT_REPORT.md        # Technical report
│   ├── FAQ.md                   # Common questions
│   ├── INDEX.md                 # Documentation index
│   └── PROJECT_STRUCTURE.md     # Structure overview
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── LICENSE                      # MIT License
└── .gitignore                   # Git configuration
```

## Dataset Features

The model uses 12 features (4 primary + 8 supporting):

| Feature | Type | Example |
|---------|------|---------|
| **Screen Time** | hours/day | 8.5 |
| **Sleep Hours** | hours/night | 6.0 |
| **Gaming Hours** | hours/day | 3.2 |
| **Notifications** | per day | 145 |
| Social Media Hours | hours/day | 3.0 |
| App Opens | per day | 115 |
| Work/Study Hours | hours/day | 3.5 |
| Weekend Screen Time | hours | 8.0 |
| Age | years | 25 |
| Gender | categorical | M/F/Other |
| Stress Level | categorical | Low/Medium/High |
| Academic Impact | categorical | Yes/No |

**Target Variable:** `addicted_label` (0 = Not Addicted, 1 = Addicted)

## Installation

### Requirements
- Python 3.8 or newer
- pip (Python package manager)

### Setup

```bash
# Clone or download the project
cd smartphone-addiction-predictor

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

Train the model once using the dataset:

```bash
python src/train.py
```

**Output:**
```
🎯 Train the Addiction Predictor
============================================================
📚 Loading the addiction data...
✅ All set! Found 7500 people's data with 16 factors
🔧 Getting the data ready...
✅ All cleaned up! Working with 12 important factors
...
✨ All done! Ready to predict!
🚀 Next: Run 'python src/predict.py' to test it out
```

Creates `models/` folder with:
- `addiction_model.pkl` - Trained Random Forest model
- `scaler.pkl` - Feature normalizer
- `features.pkl` - Feature list

### 2. Make Predictions

Make addiction predictions interactively:

```bash
python src/predict.py
```

**Interactive Session:**
```
📱 Addiction Check Tool
============================================================

📝 Tell me about your phone habits:
   (Type 'q' to quit anytime)

  📊 How many hours on phone daily? (0-24): 8
  💤 How many hours sleep per night? (0-24): 6
  🎮 How many hours gaming daily? (0-24): 3
  🔔 How many notifications daily? (0-1000): 150

============================================================
🔍 Results
============================================================

📋 What You Told Me:
   • Phone time:     8.0 hours/day
   • Sleep:          6.0 hours/night
   • Gaming:         3.0 hours/day
   • Notifications:  150/day

🔴 You Are: ⚠️  ADDICTED

📊 Confidence:
   • Healthy:   8%
   • Addicted:  92%

💡 Advice:
   ⚠️ Try cutting screen time
   ⚠️ Fix your sleep schedule
   ⚠️ Reduce games and alerts
   ⚠️ Take phone-free breaks
```

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 86.45% |
| Precision | 85.20% |
| Recall | 88.90% |
| F1-Score | 87.02% |
| ROC-AUC | 0.9412 |

**What this means:**
- **Accuracy:** Gets it right 86 out of 100 times
- **Precision:** When it says "addicted," it's correct 85% of the time
- **Recall:** Catches 89% of people who are actually addicted
- **ROC-AUC:** Excellent at separating addicted from non-addicted

## Explore with Jupyter Notebook

See how the model works under the hood:

```bash
jupyter notebook notebook/analysis.ipynb
```

This notebook shows:
- Data loading and exploration
- Feature preprocessing
- Model training
- Performance evaluation
- Model serialization

## Documentation

- **[PROJECT_REPORT.md](docs/PROJECT_REPORT.md)** - Detailed technical report
- **[FAQ.md](docs/FAQ.md)** - Frequently asked questions
- **[INDEX.md](docs/INDEX.md)** - Documentation index
- **[PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - Project structure overview

## Features

✅ Simple 4-input prediction system
✅ 86%+ accuracy on test data
✅ Random Forest classification model
✅ Feature scaling and normalization
✅ Interactive CLI prediction tool
✅ Jupyter notebook for exploration
✅ Beginner-friendly documentation
✅ Production-ready code
✅ MIT licensed

## Examples & Tips

### Example Scenarios

**Scenario 1: Healthy User**
```
Screen time: 4 hours/day
Sleep: 8 hours/night
Gaming: 1 hour/day
Notifications: 50/day
→ Prediction: NOT ADDICTED ✅
```

**Scenario 2: At-Risk User**
```
Screen time: 9 hours/day
Sleep: 5 hours/night
Gaming: 4 hours/day
Notifications: 300/day
→ Prediction: ADDICTED ⚠️
```

### Tips for Accurate Predictions

1. Provide honest estimates of your usage
2. Average values over a typical day
3. Include weekend usage in calculations
4. Consider seasonal variations

## Understanding Your Results

### Prediction Output
- 🟢 **HEALTHY** (0) = Not addicted to smartphone
- 🔴 **ADDICTED** (1) = Addicted to smartphone

### Confidence Scores
The model provides confidence percentages:
- **95% Healthy, 5% Addicted** = Very confident you're healthy
- **52% Healthy, 48% Addicted** = Model is uncertain

## What Happens Behind the Scenes

### Training (`src/train.py`)
1. Loads 7,500 people's phone usage data from `data/data.csv`
2. Cleans and preprocesses the data (handles missing values, encoding)
3. Selects 12 best features, trains the Random Forest model
4. Evaluates performance and serializes model artifacts to `models/`

## Next Steps

- Modify input prompts in `src/predict.py` for your specific needs
- Adjust model hyperparameters in `src/train.py` to improve accuracy
- Explore the dataset in `notebook/analysis.ipynb` to understand patterns
- See [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) for technical details and deployment guide

## Quick Command Reference

| Task | Command |
|------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Train the model | `python src/train.py` |
| Make predictions | `python src/predict.py` |
| Explore data | `jupyter notebook notebook/analysis.ipynb` |

## Technology Stack

- **Python 3.8+** - Programming language
- **scikit-learn** - Machine learning
- **pandas** - Data processing
- **numpy** - Numerical computing
- **joblib** - Model serialization
- **Jupyter** - Interactive notebooks

## License

MIT License - Free to use, modify, and distribute. See [LICENSE](LICENSE)

## Project Info

- **Type:** Machine Learning Classification
- **Status:** Production Ready
- **Version:** 1.0
- **Created:** March 2026
- **Dataset:** 7,500 user profiles
- **Model:** Random Forest Classifier

## Questions?

- Check [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) for technical deep dives
- Search [docs/FAQ.md](docs/FAQ.md) for common questions
- Open an issue on GitHub for bug reports

