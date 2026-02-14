from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

class ConvLEMWildfire(nn.Module):

    """
    ConvLEM-based wildfire prediction model (minimal version).
    """
    
    def __init__(
        self,
        in_dim: int,
        num_counties: int,
        past_days: int,
        hidden_dim: int = 144,
        num_layers: int = 2,
        dt: float = 1.0,
        activation: str = 'tanh',
        use_reset_gate: bool = False,
        dropout: float = 0.1,
        adjacency: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.num_counties = num_counties
        self.past_days = past_days
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Placeholder: just a simple linear layer for now
        self.placeholder = nn.Linear(in_dim, 1)
    
    def forward(self, x: torch.Tensor, adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Minimal forward pass.
        
        Args:
            x: (batch, past_days, num_counties, in_dim)
            adjacency: Optional adjacency matrix
            
        Returns:
            logits: (batch, num_counties)
        """
        B, T, N, F = x.shape
        
        # Validate shapes
        if T != self.past_days:
            raise ValueError(f"Expected past_days={self.past_days}, got {T}")
        if N != self.num_counties:
            raise ValueError(f"Expected num_counties={self.num_counties}, got {N}")
        if F != self.in_dim:
            raise ValueError(f"Expected in_dim={self.in_dim}, got {F}")
        
        # Placeholder forward: just use last timestep
        last_step = x[:, -1, :, :]  # (B, N, F)
        logits = self.placeholder(last_step).squeeze(-1)  # (B, N)
        
        return logits

		
"""
Builder function for ConvLEM wildfire prediction model.
This builder follows PyHazards conventions:
- Takes task + required params + **kwargs
- Validates task type
- Returns instantiated model with merged defaults

    Args:
        task: Task type (must be 'classification')
        in_dim: Number of input features per county-day
        num_counties: Number of counties (graph nodes)
        past_days: Number of historical days in input sequence
        **kwargs: Additional hyperparameters:
            - hidden_dim: Hidden state dimension (default: 144)
            - num_layers: Number of encoder/decoder layer pairs (default: 2)
            - dt: Time step parameter for LEM mechanism (default: 1.0)
            - activation: Activation function 'tanh' or 'relu' (default: 'tanh')
            - use_reset_gate: Use reset gate variant (default: False)
            - dropout: Dropout rate (default: 0.1)
            - adjacency: Optional adjacency matrix (N, N) (default: None)
    
    Returns:
        Instantiated ConvLEMWildfire model
    
    Raises:
        ValueError: If task is not 'classification'
    """
		    
def convlem_wildfire_builder(
    task: str,
    in_dim: int,
    num_counties: int,
    past_days: int,
    **kwargs,
) -> ConvLEMWildfire:
    """Builder function for ConvLEM wildfire model."""
    
    if task.lower() not in {"classification", "binary_classification"}:
        raise ValueError(
            f"ConvLEM wildfire model is classification-only, got task='{task}'"
        )
    
    return ConvLEMWildfire(
        in_dim=in_dim,
        num_counties=num_counties,
        past_days=past_days,
        hidden_dim=kwargs.get("hidden_dim", 144),
        num_layers=kwargs.get("num_layers", 2),
        dt=kwargs.get("dt", 1.0),
        activation=kwargs.get("activation", "tanh"),
        use_reset_gate=kwargs.get("use_reset_gate", False),
        dropout=kwargs.get("dropout", 0.1),
        adjacency=kwargs.get("adjacency"),
    )