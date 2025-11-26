# EMS Dispatch Optimization

This project analyzes EMS dispatch data from Allegheny County using both supervised and unsupervised machine learning techniques.

## Structure

```
EMS_Dispatch_Optimization/
│
├── data/
│   └── allegheny_county_911_EMS_dispatches.csv
│
├── src/
│   ├── supervised/
│   │   └── supervised_models.py
│   ├── unsupervised/
│   │   └── unsupervised_models.py
│   └── main.py
│
└── notebooks/
    └── analysis_clean.ipynb
```

## How to Run

### Supervised Models
```
python src/supervised/supervised_models.py
```

### Unsupervised Models
```
python src/unsupervised/unsupervised_models.py
```

### Full Pipeline
```
python src/main.py
```
