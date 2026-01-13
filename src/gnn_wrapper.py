import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin

# Import our custom PyTorch model
from gnn_model import LSTM_GAT_Model
# Import the function to build the graph 
from graph_nn.graph_def import compute_adj_matrix_based_on_correlation

class GNNRegressor(BaseEstimator, RegressorMixin):
    """
    The GNNRegressor acts as a 'Wrapper' (or Adapter).
    
    Why is this file necessary?
    ---------------------------
    1. The existing project infrastructure (backtester) expects a Scikit-Learn model 
       (with .fit() and .predict() methods).
    2. Our Deep Learning model is written in PyTorch (with tensors, epochs, gradients).
    
    This class translates the backtester's commands into PyTorch commands.
    It also handles the 'Dynamic' aspect by rebuilding the graph at every training step.
    """

    def __init__(self, window_size=3, hidden_dim=32, num_heads=2, epochs=50, lr=0.01, corr_threshold=0.5, loss=nn.MSELoss()):
        """
        Hyperparameters for the model and training process.
        """
        self.window_size = window_size      # Lookback period (e.g., 20 days) for the LSTM
        self.hidden_dim = hidden_dim        # Size of the internal neural network layers
        self.num_heads = num_heads          # Number of attention heads for the GAT
        self.epochs = epochs                # How many times we iterate over the data
        self.lr = lr                        # Learning rate (speed of learning)
        self.corr_threshold = corr_threshold # Minimum correlation to create an edge in the graph
        
        self.model = None                   # The actual PyTorch model
        self.edge_index = None              # The graph structure
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = loss

    def _prepare_graph(self, X_df):
        """
        Internal method to construct the Dynamic Graph.
        
        Logic:
        At each training interval (e.g., every year), we look at the recent data 
        and calculate the correlation matrix. If two stocks are highly correlated 
        (> threshold), we create an edge between them.
        
        This satisfies the "Dynamic Graph" requirement of the project.
        """
        # Calculate Spearman correlation (robust to outliers)
        X_df = pd.DataFrame(X_df)
        corr_df = X_df.corr(method="spearman")
        
        # Remove self-loops (correlation of 1 with itself)
        np.fill_diagonal(corr_df.values, 0)
        
        # Create adjacency matrix: 1 if correlated, 0 otherwise
        adj_matrix = np.where(np.abs(corr_df.values) >= self.corr_threshold, 1, 0)
        
        # Convert to PyTorch Geometric format (Edge Index: [2, Num_Edges])
        rows, cols = np.where(adj_matrix == 1)
        return torch.tensor([rows, cols], dtype=torch.long).to(self.device)

    def _prepare_tensors(self, X, y=None):
        """
        Internal method to transform 2D DataFrames (Time x Stocks)
        into 3D PyTorch Tensors (Samples x Stocks x Window_Size).

        Why? The LSTM needs a sequence of past data, not just the current value.
        """
        X = pd.DataFrame(X)
        if y is not None:
            y = pd.DataFrame(y)

        data = X.values
        X_list, y_list = [], []

        # Determine the loop range
        end_idx = len(data) if y is None else len(data)

        # Sliding Window Logic
        for t in range(self.window_size, end_idx):
            # Input: Data from (t - window) to (t)
            X_list.append(data[t - self.window_size:t, :])

            # Target: Data at (t) - only during training
            if y is not None:
                y_list.append(y.iloc[t].values)

        # Stack into numpy arrays
        X_tensor = np.array(X_list)  # Shape: (Samples, Window, Stocks)

        # Permute dimensions to match model expectation:
        # From (Samples, Window, Stocks) -> (Samples, Stocks, Window)
        X_tensor = np.transpose(X_tensor, (0, 2, 1))

        # Add feature dimension (Samples, Stocks, Window, 1)
        X_tensor = X_tensor[..., np.newaxis]

        # Convert to PyTorch Tensors
        if y is not None:
            return torch.FloatTensor(X_tensor).to(self.device), torch.FloatTensor(np.array(y_list)).to(self.device)
        return torch.FloatTensor(X_tensor).to(self.device)

    def fit(self, X, y, ret):
        """
        Standard Scikit-Learn 'fit' method.
        This is called by the Backtester to train the model on historical data.
        """
        # 1. Build the Graph (Dynamically based on current X)
        self.edge_index = self._prepare_graph(ret)
        self.num_stocks = ret.shape[1]
        
        # 2. Prepare Data (Sliding Window)
        X_train, y_train = self._prepare_tensors(X, y)

        # 3. Initialize the PyTorch Model
        self.model = LSTM_GAT_Model(
            num_features= 1,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=0.2
        ).to(self.device)
        
        # 4. Training Loop (Standard PyTorch procedure)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = self.loss
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            total_loss = 0

            for t in range(X_train.shape[0]):
                x_t = X_train[t]
                y_t = y_train[t]
            # Forward pass: LSTM -> GAT -> Prediction
            # Note: We pass the full batch for simplicity here.
            # For massive datasets, we would need mini-batches.
                out = self.model(x_t, self.edge_index)
                loss_t = criterion(out, y_t)
                total_loss += loss_t.item()
            
            # Backward pass
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, loss = {total_loss / X_train.shape[0]:.6f}")
            
        return self

    def predict(self, X):
        """
        Standard Scikit-Learn 'predict' method.
        This is called by the Backtester to make trading decisions.
        """
        if X.shape[0] <= self.window_size:
            raise ValueError(f"Input length {X.shape[0]} <= window_size {self.window_size}")

        self.num_stocks = X.shape[1]  # important for padding

        self.model.eval()
        X_test = self._prepare_tensors(X, y=None)  # [Samples - window_size, Stocks, Window, 1]
        if X_test.shape[0] == 0:
            return np.zeros((X.shape[0], X.shape[1]))

        preds_list = []
        with torch.no_grad():
            for t in range(X_test.shape[0]):
                x_t = X_test[t].to(self.device)
                pred_t = self.model(x_t, self.edge_index)
                preds_list.append(pred_t.cpu())

        preds = torch.stack(preds_list)  # [Samples - window_size, Stocks]

        # Pad first 'window_size' steps
        padding = np.zeros((self.window_size, self.num_stocks))
        return np.vstack([padding, preds.numpy()])