import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
from .swarm import System4Swarm

class Gemma4SafetyFilter(nn.Module):
    """
    The Gemma 4 E4B Zero-Gradient Safety Filter wrapper.
    It projects the multimodal LLM output token embeddings (2048-d)
    into the 64-dimensional OOD sensory space of the System 4 Swarm,
    runs the swarm with rolling solver warm-starting, and flags safety overrides
    whenever a Crisis regime transition is detected by the Mahalanobis detector.
    """
    def __init__(self, swarm: System4Swarm, embedding_dim: int = 2048, d_in: int = 64):
        super().__init__()
        self.swarm = swarm
        self.embedding_dim = embedding_dim
        self.d_in = d_in
        
        # Linear projection mapping Gemma 4 E4B token embeddings to Swarm inputs
        self.projection = nn.Linear(embedding_dim, d_in)
        
        # Rolling solver state for streaming warm-starts
        self.z_prev: Optional[torch.Tensor] = None

    def reset_filter(self):
        """
        Clears the warm-start fixed-point cache.
        """
        self.z_prev = None

    def forward_filter(self, token_embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any], bool]:
        """
        Processes incoming Gemma 4 token embeddings and monitors safety limits.
        
        token_embeddings: shape (B, 2048) or (B, SeqLen, 2048)
        
        Returns:
        - latent_action: shape (B, 256) Swarm constraint representation
        - info: diagnostic metadata containing the current PCAS regime
        - override_triggered: boolean flag indicating if safety overrides are active (Crisis regime)
        """
        # 1. Align dimensions: if sequential, average across sequence length (global pool)
        if token_embeddings.dim() == 3:
            embeddings_pooled = torch.mean(token_embeddings, dim=1) # (B, 2048)
        else:
            embeddings_pooled = token_embeddings # (B, 2048)
            
        # 2. Project token representation to swarm sensory dimension (64-d)
        swarm_obs = self.projection(embeddings_pooled) # (B, 64)
        
        # 3. Forward pass through System 4 Swarm with solver warm-starting
        latent_action, info = self.swarm(swarm_obs, z_prev=self.z_prev)
        
        # Keep warm-start reference for the next streaming step
        self.z_prev = info["Z_star"].detach()
        
        # 4. Detect if safety overrides must be engaged
        # active_regime == 1 represents Crisis regime (PCAS triggered)
        override_triggered = (self.swarm.active_regime.item() == 1)
        
        return latent_action, info, override_triggered

if __name__ == "__main__":
    # Sanity check wrapper
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    swarm = System4Swarm(d_in=64, d=256, gamma=0.98, warm_up_steps=10).to(device)
    # Calibrate dummy monitor
    swarm.is_calibrated.copy_(torch.tensor(True, dtype=torch.bool, device=device))
    
    filter_wrapper = Gemma4SafetyFilter(swarm).to(device)
    
    # Simulate a batch of Gemma 4 multimodal embeddings (Batch=2, Dim=2048)
    dummy_embeddings = torch.randn(2, 2048).to(device)
    
    out, info, override = filter_wrapper.forward_filter(dummy_embeddings)
    print("Filter Wrapper Sanity Check:")
    print(f" - Latent Action shape: {out.shape}")
    print(f" - Active PCAS Regime: {info['regime']}")
    print(f" - Safety Override Engaged: {override}")
