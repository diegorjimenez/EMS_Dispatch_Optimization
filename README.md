# EMS Dispatch Optimization

This project analyzes EMS dispatch data from Allegheny County using both supervised and unsupervised machine learning techniques.

## Structure

```
EMS_Dispatch_Optimization/
│
├── data/
│   └── allegheny_county_911_EMS_dispatches_sampledata_5k.csv
│
├── src/
│   ├── supervised/
│   │   └── supervised_models_drive.py
│   ├── unsupervised/
│   │   └── unsupervised_models_drive.py
│   └── main.py
│
└── notebooks/
    └── full_analysis_.ipynb
```

## How to Run

### Supervised Models
```
python src/supervised/supervised_models_drive.py
```

### Unsupervised Models
```
python src/unsupervised/unsupervised_models_drive.py
```

### Full Pipeline
```
python src/main.py
```
