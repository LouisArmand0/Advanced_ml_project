# Graph-Based Systematic Trading: Dynamic Graph Neural Networks for Stock Prediction and Portfolio Construction
Welcome on our repo for the Advanced Machine Learning project where we developed a systematic cross-sectional momentum strategy on the US equity market. 

### Background

Our approach is based on the methods described in [Pacreau et al., 2021](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3976168). 
We extend their work by combining GNNs with factor-based momentum strategies and providing a complete pipeline from raw data to backtesting.

### Installation

Clone the repo and install dependencies:

```
git clone https://github.com/LouisArmand0/Advanced_ml_project.git
cd Advanced_ml_project
pip install -r requirements.txt
````

### Running the project

1. Generate raw data:
Run the following command in your terminal:
```
python data/raw_data_generation.py
```

2. Run a backtest

```
python hyperparameter_search.py
```
