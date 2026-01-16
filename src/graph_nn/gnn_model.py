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
    
    def __init__(self, num_features, hidden_dim, num_layers_lstm, num_heads, dropout):
        """
        Initialize the model layers.

        Args:
            num_features (int): number of input features per time step (e.g., 1 for just returns).
            hidden_dim (int): size of the internal embedding vector.
            num_heads (int): number of attention heads for the GAT (multi-view learning).
            dropout (float): dropout rate to prevent overfitting.
        """
        super(LSTM_GAT_Model, self).__init__()
        

        # Temporal Layer (LSTM) 
        self.lstm = nn.LSTM(
            input_size=num_features, 
            hidden_size=hidden_dim, 
            num_layers=num_layers_lstm,
            batch_first=True
        )
        

        # Spatial Layer (Graph Attention Network) 
        self.gat = GATv2Conv(
            in_channels=hidden_dim,    
            out_channels=hidden_dim,   
            heads=num_heads,            
            concat=False,          
            dropout=dropout
        )
        

        # Prediction Layer 
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
        

        # Temporal embedding (LSTM)
        _, (h_n, _) = self.lstm(x)
        temporal_embeddings = h_n[-1]
        

        # Spatial aggregation (GAT)
        spatial_embeddings = self.gat(
            temporal_embeddings, 
            edge_index, 
            edge_attr=edge_weight
        )
        
        # Non-linear activation function (ELU is standard for GATs)
        spatial_embeddings = F.elu(spatial_embeddings)
        
        # Regularization (dropout) to force robustness
        spatial_embeddings = self.dropout(spatial_embeddings)
        
 
        # Final prediction
        prediction = self.predictor(spatial_embeddings)
        
        return prediction.squeeze()