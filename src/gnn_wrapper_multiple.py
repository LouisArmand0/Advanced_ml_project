import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin

# Import our custom PyTorch model
from gnn_model import LSTM_GAT_Model
# Import the function to build the graph 
from graph_nn.graph_def import compute_adj_matrix_based_on_correlation

class GNNRegressor_Multiple(BaseEstimator, RegressorMixin):
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

    def __init__(self, window_size: int, hidden_dim: int, num_layers_lstm: int,  num_heads: int, epochs: int, lr: float, corr_threshold: float,
                 nb_features_per_stock: int, drop_out: float,  loss=nn.MSELoss()):
        """
        Hyperparameters for the model and training process.
        """
        self.window_size = window_size      # Lookback period (e.g., 20 days) for the LSTM
        self.hidden_dim = hidden_dim        # Size of the internal neural network layers
        self.num_heads = num_heads          # Number of attention heads for the GAT
        self.epochs = epochs                # How many times we iterate over the data
        self.lr = lr                        # Learning rate (speed of learning)
        self.corr_threshold = corr_threshold # Minimum correlation to create an edge in the graph
        self.loss = loss
        self.nb_features_per_stock = nb_features_per_stock
        self.drop_out = drop_out
        self.num_layers_lstm = num_layers_lstm
        
        self.model = None                   # The actual PyTorch model
        self.edge_index = None              # The graph structure
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        X = pd.DataFrame(X)
        if y is not None:
            y = pd.DataFrame(y)

        T = len(X)
        n_features = self.nb_features_per_stock
        n_stocks = X.shape[1] // n_features

        # (Time, Stocks, Features)
        data = X.values.reshape(T, n_stocks, n_features)

        X_list, y_list = [], []

        for t in range(self.window_size, T):
            # (Window, Stocks, Features)
            window = data[t - self.window_size:t]

            X_list.append(window)

            if y is not None:
                y_list.append(y.iloc[t].values)

        # (Samples, Window, Stocks, Features)
        X_tensor = np.array(X_list)

        # → (Samples, Stocks, Window, Features)
        X_tensor = np.transpose(X_tensor, (0, 2, 1, 3))

        if y is not None:
            return (
                torch.FloatTensor(X_tensor).to(self.device),
                torch.FloatTensor(np.array(y_list)).to(self.device)
            )

        return torch.FloatTensor(X_tensor).to(self.device)

    def fit(self, X, y, ret, X_val=None, y_val=None):
        """
        Standard Scikit-Learn 'fit' method.
        This is called by the Backtester to train the model on historical data.
        """
        # 1. Build the Graph (Dynamically based on current X)
        self.edge_index = self._prepare_graph(ret)
        self.num_stocks = ret.shape[1]
        
        # 2. Prepare Data (Sliding Window)
        X_train, y_train = self._prepare_tensors(X, y)
        if X_val is not None and y_val is not None:
            X_val, y_val = self._prepare_tensors(X_val, y_val)

        # 3. Initialize the PyTorch Model
        self.model = LSTM_GAT_Model(
            num_features= self.nb_features_per_stock, #nb of features per stock
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.drop_out,
            num_layers_lstm=self.num_layers_lstm,
        ).to(self.device)
        
        # 4. Training Loop (Standard PyTorch procedure)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = self.loss

        train_losses = []
        val_losses = []

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            total_train_loss = 0

            for t in range(X_train.shape[0]):
                x_t = X_train[t].to(self.device)
                y_t = y_train[t].to(self.device)

                # Forward
                out = self.model(x_t, self.edge_index)
                loss = criterion(out, y_t)
                total_train_loss += loss.item()

                # Backward
                loss.backward()

            optimizer.step()

            # Average training loss
            avg_train_loss = total_train_loss / X_train.shape[0]
            train_losses.append(avg_train_loss)

            # --- Validation loss ---
            if X_val is not None and y_val is not None:
                self.model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for t in range(X_val.shape[0]):
                        x_val_t = X_val[t].to(self.device)
                        y_val_t = y_val[t].to(self.device)

                        val_out = self.model(x_val_t, self.edge_index)
                        val_loss = criterion(val_out, y_val_t)
                        total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / X_val.shape[0]
                val_losses.append(avg_val_loss)

                self.model.train()  # Back to training mode

            # Optional logging every 10 epochs
            if epoch % 10 == 0:
                if X_val is not None:
                    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
                else:
                    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}")
        self.train_losses = train_losses
        self.val_losses = val_losses
        return self

    def predict(self, X):
        """
        Standard Scikit-Learn 'predict' method.
        This is called by the Backtester to make trading decisions.
        """
        if X.shape[0] <= self.window_size:
            raise ValueError(f"Input length {X.shape[0]} <= window_size {self.window_size}")

        self.model.eval()
        X_test = self._prepare_tensors(X, y=None)  # [Samples - window_size, Stocks, Window, 1]
        if X_test.shape[0] == 0:
            return np.zeros((X.shape[0], X.shape[1]))
        self.num_stocks = X_test.shape[1]

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