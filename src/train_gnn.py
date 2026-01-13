import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# IMPORTS 
# We import the custom model architecture we defined
from gnn_model import LSTM_GAT_Model

# We reuse the data infrastructure built by your teammate
# ensuring consistency with the rest of the project
from utils.features import (
    getting_data_for_ticker_list, 
    getting_trading_universe, 
    compute_returns
)
# We import the graph construction logic
from graph_nn.graph_def import compute_adj_matrix_based_on_correlation

# CONFIGURATION (HYPERPARAMETERS)
# WINDOW_SIZE: The LSTM looks at the past 20 days to understand the trend.
WINDOW_SIZE = 20      
# HIDDEN_DIM: The size of the internal vector representing a stock (embedding size).
HIDDEN_DIM = 32       
# NUM_HEADS: Multi-head attention allows the model to view relationships from different angles.
NUM_HEADS = 2         
# LEARNING_RATE: How fast the model updates its weights.
LEARNING_RATE = 0.001
# EPOCHS: Number of times the model sees the entire dataset.
EPOCHS = 100
# CORRELATION_THRESHOLD: Minimum correlation required to create an edge between two stocks.
CORRELATION_THRESHOLD = 0.5 

def prepare_data():
    """
    Step 1: Load Raw Data and Compute Returns.
    This function leverages the existing 'features.py' utility to fetch price data
    and transform it into log-returns, which are stationary and better for DL models.
    """
    print("1. Loading Data...")
    
    # Get the list of all available stocks
    trading_universe = getting_trading_universe()
    tickers = trading_universe['symbol'].to_list()
    
    # NOTE: We take a subset (e.g., first 50 stocks) for faster training/testing (juste pour tester le code)
    tickers = [t for t in tickers if t != 'SPY'][:50] 
    
    # Fetch historical prices using the teammate's function
    df_prices = getting_data_for_ticker_list(tickers)
    
    print("2. Computing Log-Returns...")
    # Compute returns using the teammate's function
    df_returns = compute_returns(df_prices, return_type='log')
    
    # Pivot the table to have a matrix format: Rows=Dates, Columns=Stocks
    # This format is required for matrix operations in PyTorch
    returns_matrix = df_returns.pivot(index='date', columns='stock_name', values='log_ret')
    returns_matrix = returns_matrix.dropna() # Remove missing values to avoid NaNs
    
    return returns_matrix

def create_sliding_windows(data_matrix, window_size):
    """
    Step 2: Time-Series Preprocessing (Sliding Window).
    
    LSTMs cannot take a single point in time; they need a sequence.
    We transform the (Time, Stocks) matrix into a 3D Tensor:
    (Samples, Window_Size, Stocks).
    
    Logic:
    To predict the return at day T, we use returns from [T-20 ... T-1].
    """
    print("3. Creating Sliding Windows (Temporal Sequences)...")
    data = data_matrix.values # Convert Pandas DataFrame to Numpy Array
    X, Y = [], []
    
    # Iterate through time to create sequences
    for t in range(window_size, len(data)):
        # Input: Sequence of past 'window_size' days
        x_window = data[t-window_size:t, :] 
        # Target: Return at the current day 't'
        y_target = data[t, :]
        
        X.append(x_window)
        Y.append(y_target)
        
    X = np.array(X) # Shape: (Samples, Window, N_Nodes)
    Y = np.array(Y) # Shape: (Samples, N_Nodes)
    
    # Transpose dimensions to match PyTorch LSTM expectation:
    # From (Samples, Window, Nodes) -> (Samples, Nodes, Window)
    # We add a last dimension '1' because we have 1 feature (Return)
    X = np.transpose(X, (0, 2, 1)) 
    X = X[..., np.newaxis]         # Shape: (Samples, N_Nodes, Window, 1)
    
    return torch.FloatTensor(X), torch.FloatTensor(Y)

def get_graph_structure(returns_matrix, threshold):
    """
    Step 3: Graph Construction.
    
    Following the Amundi paper, we define the market structure based on correlation.
    If two stocks have a correlation > threshold, they are connected (Edge = 1).
    This allows the GAT to aggregate information from correlated assets.
    """
    print("4. Building Correlation Graph...")
    
    # We use the first year (252 days) to estimate the initial correlation structure
    train_slice = returns_matrix.iloc[:252] 
    
    # Reuse the logic defined in 'graph_def.py'
    adj_matrix, _ = compute_adj_matrix_based_on_correlation(train_slice, threshold=threshold)
    
    # Convert Adjacency Matrix to Edge Index format (COO) required by PyTorch Geometric
    # rows, cols contains the indices of connected nodes
    rows, cols = np.where(adj_matrix == 1)
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    
    print(f"   -> Graph created with {edge_index.shape[1]} edges.")
    return edge_index

if __name__ == "__main__":


    # PHASE A: DATA PREPARATION
    returns_matrix = prepare_data()
    
    # Build the static graph (for this training demo)
    edge_index = get_graph_structure(returns_matrix, CORRELATION_THRESHOLD)
    
    # Create temporal sequences
    X_tensor, Y_tensor = create_sliding_windows(returns_matrix, WINDOW_SIZE)
    
    # Split into Train (80%) and Test (20%) sets to validate learning
    split_idx = int(0.8 * len(X_tensor))
    X_train, Y_train = X_tensor[:split_idx], Y_tensor[:split_idx]
    X_test, Y_test = X_tensor[split_idx:], Y_tensor[split_idx:]
    
    print(f"   -> Train shape: {X_train.shape}, Test shape: {X_test.shape}")


    # PHASE B: MODEL INITIALIZATION
    # Use GPU (cuda) or Apple Silicon (mps) if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LSTM_GAT_Model(
        num_features=1,       # Input is just returns
        hidden_dim=HIDDEN_DIM,# Size of embedding
        num_heads=NUM_HEADS,  # Multi-head attention
        dropout=0.2           # Regularization
    ).to(device)
    
    # Optimizer: Adam is standard for Deep Learning
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Loss Function: Mean Squared Error (Standard for regression problems)
    criterion = nn.MSELoss() 

    # Move graph structure to the computing device
    edge_index = edge_index.to(device)
    

    # PHASE C: TRAINING LOOP
    print("\nStarting Training Loop...")
    train_losses = []
    
    model.train() # Set model to training mode (enables Dropout)
    
    for epoch in range(EPOCHS):
        # We use Stochastic Gradient Descent (SGD) logic here
        # We sample a random batch of 32 days to update the weights
        indices = torch.randperm(X_train.size(0))[:32]
        batch_loss = 0
        
        optimizer.zero_grad() # Reset gradients
        
        for i in indices:
            x_day = X_train[i].to(device) # Input sequence
            y_day = Y_train[i].to(device) # Target return
            
            # Forward Pass: Predict returns
            out = model(x_day, edge_index)
            
            # Compute Error
            loss = criterion(out, y_day)
            batch_loss += loss
        
        # Average loss over the batch
        batch_loss = batch_loss / 32
        
        # Backward Pass: Compute gradients and update weights
        batch_loss.backward()
        optimizer.step()
        
        train_losses.append(batch_loss.item())
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {batch_loss.item():.6f}")


    # PHASE D: VISUALIZATION
    # Plotting the loss curve to verify convergence
    # If the curve goes down, the model is learning!
    plt.plot(train_losses)
    plt.title("Training Loss Curve (MSE)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    
    print("Training complete! Model is ready for the Backtester.")