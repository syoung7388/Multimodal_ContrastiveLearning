# Multimodal ContrastiveLearning Classification 

## Requirements
```
pip install -r requirements.txt
```

## Usage

0. Data Preparation
- you should probably download the data (https://dacon.io/competitions/official/235978/overview/description) 
- Save the downloaded data to "datasets" folder


1. Contrastive model 
```
python3 contrastive.py --classification_layer 1 
```

2. Ensemble model 
```
python3 ensemble.py --ensemble 1  
```

3. Single model 
```
python3 ensemble.py --ensemble 0 
```

