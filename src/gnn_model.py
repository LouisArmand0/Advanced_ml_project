import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class LSTM_GAT_Model(nn.Module):
    """
    Model for stock trend forecasting based on the Amundi Paper (Pacreau et al., 2021).
    
    Architecture overview:
    1. Temporal component (LSTM): extracts individual trends from historical data for each stock.
    2. Spatial component (GAT): aggregates information from related stocks (neighbors in the graph).
    3. Prediction component (Linear): outputs the expected return.
    """
    
    def __init__(self, num_features, hidden_dim, num_heads=2, dropout=0.2):
        """
        Initialize the model layers.

        Args:
            num_features (int): number of input features per time step (e.g., 1 for just returns).
            hidden_dim (int): size of the internal embedding vector.
            num_heads (int): number of attention heads for the GAT (multi-view learning).
            dropout (float): dropout rate to prevent overfitting.
        """
        super(LSTM_GAT_Model, self).__init__()
        
        # --- 1. Temporal Layer (LSTM) ---
        # The LSTM processes the time-series of each stock independently.
        # It captures 'Momentum' and 'Mean reversion' patterns.
        # input_shape: (Batch, Sequence_Length, Features)
        self.lstm = nn.LSTM(
            input_size=num_features, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        # --- 2. Spatial Layer (Graph Attention Network) ---
        # The GAT allows stocks to 'talk' to each other based on the graph structure (Edge Index).
        # It captures 'Sector effects' and 'Market Contagion'.
        # We use GATv2Conv (more expressive than standard GAT).
        self.gat = GATv2Conv(
            in_channels=hidden_dim,     # Input is the output of the LSTM
            out_channels=hidden_dim,    # Output size
            heads=num_heads,            # Number of parallel attention mechanisms
            concat=False,          
            dropout=dropout
        )
        
        # --- 3. Prediction Layer ---
        # A simple linear regression to map the final embedding to a return prediction.
        self.dropout = nn.Dropout(dropout)
        self.predictor = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass of the neural network.

        Args:
            x (Tensor): input data of shape [Num_Stocks, Window_Size, Num_Features].
            edge_index (Tensor): graph connectivity [2, Num_Edges].
            edge_weight (Tensor, optional): strength of connections.

        Returns:
            Tensor: predicted return for each stock [Num_Stocks].
        """
        

        # TEMPORAL EMBEDDING (LSTM)
        # We pass the history of each stock through the LSTM.
        # We only care about the last hidden state (h_n), which summarizes the whole window.
        # self.lstm returns: output, (h_n, c_n)
        _, (h_n, _) = self.lstm(x)
        
        # h_n shape is [1, Num_Stocks, Hidden_Dim]. We remove the first dimension.
        # 'temporal_embeddings' represents the individual state of each stock.
        temporal_embeddings = h_n.squeeze(0) 
        

        # SPATIAL AGGREGATION (GAT)
        # Now, stocks exchange information with their neighbors defined in 'edge_index'.
        # (if Apple and Microsoft are connected, Apple will update its state based on Microsoft's state)
        spatial_embeddings = self.gat(
            temporal_embeddings, 
            edge_index, 
            edge_attr=edge_weight
        )
        
        # Non-linear activation function (ELU is standard for GATs)
        spatial_embeddings = F.elu(spatial_embeddings)
        
        # Regularization (dropout) to force robustness
        spatial_embeddings = self.dropout(spatial_embeddings)
        
 
        # STEP 3: FINAL PREDICTION
        # Project the enriched embeddings to a single scalar (expected return).
        prediction = self.predictor(spatial_embeddings)
        
        # Remove extra dimensions to get a flat vector of size [Num_Stocks]
        return prediction.squeeze()