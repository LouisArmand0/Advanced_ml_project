# Graph-Based Systematic Trading: Dynamic Graph Neural Networks for Stock Prediction and Portfolio Construction
Welcome on our repo for the Advanced Machine Learning project where we developed a systematic cross-sectional momentum strategy on the US equity market. 

### Background

Our approach is based on the methods described in [Pacreau et al., 2021](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3976168). 
We extend their work by combining GNNs with factor-based momentum strategies and providing a complete pipeline from raw data to backtesting.

### Structure of the Repo

- **`graph_nn/`**  
  Contains all scripts related to the creation of the adjacency matrix,  
  the model (`.pt` Torch file), and the GNN wrapper that makes the model compatible with `sklearn`'s `fit`/`predict` framework.

- **`utils/`**  
  Helper functions used throughout the project:  
  - `backtesting.py`: Implementation of the Walk-Forward backtest. The model is trained on Y1 and tested on Y2, then rolled forward.  
  - `custom_loss.py`: Implementation of the Negative Sharpe Loss.  
  - `features.py`: Functions for feature computation.  
  - `mv_estimator.py`: Vectorized implementation of Mean-Variance portfolio optimization.

- **Main script**  
  `hyperparameter.py` is the main entry point. It trains models with different hyperparameter sets and compares their performance using the Walk-Forward backtest.


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
