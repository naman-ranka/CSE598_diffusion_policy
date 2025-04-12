import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization as used in advanced diffusion models like Imagen.
    
    This implementation modulates the normalization parameters (scale and shift)
    based on timestep embeddings, allowing the normalization to adapt to the
    current denoising stage.
    """
    
    def __init__(self, feature_dim, time_emb_dim=None, eps=1e-5):
        """
        Initialize AdaLayerNorm.
        
        Args:
            feature_dim: Dimension of the input features to normalize
            time_emb_dim: Dimension of the time embedding input to the MLP.
                           If None, use feature_dim.
            eps: Small constant for numerical stability
        """
        super().__init__()
        
        # Layer norm without affine parameters (since we'll generate them adaptively)
        self.norm = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=eps)
        
        # If time_emb_dim is not specified, use feature_dim
        if time_emb_dim is None:
            time_emb_dim = feature_dim
            
        # MLP to project time embeddings to modulation parameters
        # Ensure the input dimension matches the time embedding dimension
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, 4 * feature_dim), # Input dim is time_emb_dim
            nn.SiLU(),
            nn.Linear(4 * feature_dim, 2 * feature_dim)  # 2x for scale and shift
        )
        
        self.feature_dim = feature_dim
        
    def forward(self, x, time_emb):
        """
        Apply adaptive layer normalization.
        
        Args:
            x: Input tensor [batch_size, seq_len, feature_dim] or [batch_size, feature_dim]
            time_emb: Time embedding tensor [batch_size, time_emb_dim]
        
        Returns:
            Normalized tensor with same shape as input
        """
        # Handle different input shapes
        orig_shape = x.shape
        is_3d = len(orig_shape) == 3

        if is_3d:  # [batch_size, seq_len, feature_dim]
            batch_size, seq_len, _ = orig_shape
        else:  # [batch_size, feature_dim] - Temporarily add seq_len dim
            batch_size = orig_shape[0]
            seq_len = 1
            x = x.unsqueeze(1)
        
        # Normalize the input
        # LayerNorm expects input shape [*, normalized_shape]
        # For [B, T, C], normalized_shape is [C]
        # For [B, C], normalized_shape is [C]
        x_norm = self.norm(x) # Output shape [B, T, C] or [B, 1, C]
        
        # Project time embeddings to scale and shift
        # time_emb shape is expected to be [B, time_emb_dim]
        time_emb_proj = self.time_mlp(time_emb)  # [B, 2*feature_dim]
        
        # Split into scale and shift
        scale, shift = time_emb_proj.chunk(2, dim=-1)  # Each is [B, feature_dim]
        
        # Add sequence dimension for broadcasting: [B, 1, feature_dim]
        scale = scale.unsqueeze(1) 
        shift = shift.unsqueeze(1) 
        
        # Apply adaptive modulation: scale * norm(x) + shift
        # Broadcasting rules:
        # x_norm: [B, T, C] or [B, 1, C]
        # scale:  [B, 1, C]
        # shift:  [B, 1, C]
        # Result: [B, T, C] or [B, 1, C]
        output = scale * x_norm + shift
        
        # Return with original shape
        if not is_3d:
            output = output.squeeze(1) # Back to [B, C]
            
        return output


class TransformerAdaLayerNorm(nn.Module):
    """
    Custom version of AdaLayerNorm designed to be a drop-in replacement
    for PyTorch's LayerNorm in transformer layers.
    
    In PyTorch's TransformerEncoderLayer and TransformerDecoderLayer,
    the norm is applied without any explicit timestep conditioning.
    This class provides a way to store the timestep embeddings globally
    and use them during the forward pass.
    """
    def __init__(self, feature_dim, eps=1e-5):
        super().__init__()
        self.ada_layer_norm = AdaLayerNorm(feature_dim, eps=eps)
        self._current_time_emb = None
        
    def set_timestep_embedding(self, time_emb):
        """Set the current timestep embedding to use in forward pass"""
        self._current_time_emb = time_emb
        
    def forward(self, x):
        """
        Forward pass that uses the stored timestep embedding.
        This matches PyTorch's LayerNorm interface.
        """
        if self._current_time_emb is None:
            # Fallback to standard layer norm if no timestep is set
            return nn.functional.layer_norm(
                x, 
                normalized_shape=x.shape[-1:], 
                eps=self.ada_layer_norm.norm.eps
            )
        
        return self.ada_layer_norm(x, self._current_time_emb)
