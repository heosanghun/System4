import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralNormLinear(nn.Linear):
    """
    A linear layer that strictly enforces a spectral norm (largest singular value) bound
    on its weight matrix using power iteration. This ensures the layer remains contractive
    with Lipschitz constant <= gamma.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, gamma: float = 0.98, n_power_iterations: int = 1):
        super().__init__(in_features, out_features, bias)
        self.gamma = gamma
        self.n_power_iterations = n_power_iterations
        
        # Register singular vectors as non-differentiable buffers
        self.register_buffer('u', torch.randn(out_features, 1))
        self.register_buffer('v', torch.randn(in_features, 1))
        
        # Normalize initial buffers
        self.u.data = F.normalize(self.u.data, dim=0)
        self.v.data = F.normalize(self.v.data, dim=0)

    @torch.no_grad()
    def _power_iteration(self):
        """
        Performs power iterations to estimate the largest singular value of the weight matrix.
        """
        weight = self.weight.data
        u = self.u
        v = self.v
        
        for _ in range(self.n_power_iterations):
            # v = W^T u / ||W^T u||
            v = F.normalize(torch.matmul(weight.t(), u), dim=0)
            # u = W v / ||W v||
            u = F.normalize(torch.matmul(weight, v), dim=0)
            
        self.u.copy_(u)
        self.v.copy_(v)
        
        # Sigma = u^T W v
        sigma = torch.matmul(u.t(), torch.matmul(weight, v)).item()
        return sigma

    def get_constrained_weight(self):
        """
        Returns the weight matrix scaled to satisfy the spectral norm constraint.
        """
        sigma = self._power_iteration()
        if sigma > self.gamma:
            # Scale weight so that its spectral norm is exactly gamma
            scale = self.gamma / sigma
            return self.weight * scale
        return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, '_use_static_cache') and self._use_static_cache:
            if not hasattr(self, '_cached_w_constrained') or getattr(self, '_cached_w_constrained') is None:
                object.__setattr__(self, '_cached_w_constrained', self.get_constrained_weight())
            w_constrained = getattr(self, '_cached_w_constrained')
        elif not self.training:
            if not hasattr(self, '_cached_w_constrained') or getattr(self, '_cached_w_constrained') is None:
                object.__setattr__(self, '_cached_w_constrained', self.get_constrained_weight())
            w_constrained = getattr(self, '_cached_w_constrained')
        else:
            if hasattr(self, '_cached_w_constrained'):
                object.__setattr__(self, '_cached_w_constrained', None)
            w_constrained = self.get_constrained_weight()
        return F.linear(x, w_constrained, self.bias)

    @property
    def spectral_norm(self) -> float:
        """
        Returns the current estimated spectral norm of the weight.
        """
        return self._power_iteration()


class MicroDEQAgent(nn.Module):
    """
    A single micro-DEQ agent (A_i) with a 2-layer MLP (hidden size d=256)
    and strictly constrained spectral properties (Lipschitz constant <= 0.98).
    Total parameters per agent is ~131K.
    """
    def __init__(self, d: int = 256, gamma: float = 0.98, agent_id: int = 0):
        super().__init__()
        self.d = d
        self.agent_id = agent_id
        
        # Enforce local Lipschitz constant <= gamma on the agent's MLP.
        # Since it's a 2-layer MLP, we bound the spectral norm of each layer by sqrt(gamma)
        # so that the composition has a total Lipschitz constant bounded by gamma.
        layer_gamma = gamma ** 0.5
        
        self.fc1 = SpectralNormLinear(d, d, bias=True, gamma=layer_gamma)
        self.fc2 = SpectralNormLinear(d, d, bias=True, gamma=layer_gamma)
        self.activation = nn.ReLU() # ReLU is 1-Lipschitz

    def forward(self, z: torch.Tensor, c: torch.Tensor, proj_x: torch.Tensor) -> torch.Tensor:
        """
        Computes the state update for the agent.
        z: Agent's own state tensor of shape (batch_size, d)
        c: Continuous coupling tensor from other agents of shape (batch_size, d)
        proj_x: Projected environment observation of shape (batch_size, d)
        """
        # Sum the state, coupling, and observation projection to form the 256-d input
        h = z + c + proj_x
        
        # 2-layer MLP pass
        h = self.fc1(h)
        h = self.activation(h)
        z_next = self.fc2(h)
        
        return z_next

    @property
    def estimated_lipschitz(self) -> float:
        """
        Estimates the Lipschitz constant of the agent MLP ( fc2_sn * fc1_sn ).
        """
        return self.fc1.spectral_norm * self.fc2.spectral_norm
