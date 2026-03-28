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
├── models/                    # Trained models (generated)
│   ├── addiction_model.pkl
│   ├── scaler.pkl
│   └── features.pkl
│
├── notebook/                  # Jupyter notebooks
│   └── analysis.ipynb
│
├── docs/                      # Documentation
│   ├── README.md             # Main guide
│   ├── PROJECT_REPORT.md     # Technical report
│   ├── FAQ.md                # Questions & answers
│   └── INDEX.md              # Documentation index
│
├── tests/                     # Unit tests
│   └── test_model.py         # Model tests
│
├── run_train.py              # Entry point: python run_train.py
├── run_predict.py            # Entry point: python run_predict.py
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── .gitignore               # Git ignore rules
```

## Quick Start

```bash
pip install -r requirements.txt
python run_train.py       # Train the model
python run_predict.py     # Make predictions
python -m pytest tests/   # Run tests
```

## Module Organization

- **src/** → Core ML logic
- **data/** → Training datasets
- **models/** → Serialized models
- **docs/** → All documentation
- **tests/** → Test suite
- **Root** → Entry points & config
