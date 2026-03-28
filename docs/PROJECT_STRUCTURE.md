# Project Structure

```
smartphone-addiction-predictor/
│
├── src/                       # Source code
│   ├── train.py              # Model training
│   └── predict.py            # Interactive predictions
│
├── data/                      # Datasets
│   └── data.csv              # 7,500 user records
│
├── models/                    # Trained models (generated after training)
│   ├── addiction_model.pkl
│   ├── scaler.pkl
│   └── features.pkl
│
├── notebook/                  # Jupyter notebooks
│   └── analysis.ipynb
│
├── docs/                      # Documentation
│   ├── PROJECT_REPORT.md     # Technical report
│   ├── FAQ.md                # Questions & answers
│   ├── INDEX.md              # Documentation index
│   └── PROJECT_STRUCTURE.md  # This file
│
├── README.md                 # Main guide (start here)
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── .gitignore               # Git ignore rules
```

## Quick Start

```bash
pip install -r requirements.txt
python src/train.py    # Train the model
python src/predict.py  # Make predictions
```

## Module Organization

- **src/** → Core ML logic
- **data/** → Training datasets
- **models/** → Serialized models
- **docs/** → All documentation
- **tests/** → Test suite
- **Root** → Entry points & config
