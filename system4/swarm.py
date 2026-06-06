import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, List

from .agent import MicroDEQAgent
from .solver import BroydenSolver

class System4Swarm(nn.Module):
    """
    The System 4 Swarm coordinates 28 micro-DEQ agents partitioned into:
    - Sensory Encoders (11 agents: 0-10)
    - Latent Reasoners (10 agents: 11-20)
    - Constraint Projectors (7 agents: 21-27)
    
    It manages:
    - Pre-Compiled Adjacency Switching (PCAS) topologies: M_normal and M_crisis.
    - A bank of 7 shared pairwise interaction matrices W_ij in R^(256 x 256) (~0.44M params).
    - Sensory observation projections P_i in R^(d_in -> 256) for sensory encoders.
    - Mahalanobis-distance anomaly detector for regime tracking.
    - Batched Broyden equilibrium solver.
    """
    def __init__(self, d_in: int = 64, d: int = 256, gamma: float = 0.98, warm_up_steps: int = 1000):
        super().__init__()
        self.d_in = d_in
        self.d = d
        self.gamma = gamma
        self.warm_up_steps = warm_up_steps
        
        # 1. 28 Micro-DEQ Agents
        self.agents = nn.ModuleList([MicroDEQAgent(d=d, gamma=gamma, agent_id=i) for i in range(28)])
        
        # 2. Sensory observation projection layers (only for the first 11 agents)
        self.proj_layers = nn.ModuleList([nn.Linear(d_in, d) for _ in range(11)])
        
        # 3. Bank of 7 shared interaction weight matrices (~0.44M parameters)
        # 7 * 256 * 256 = 458,752 parameters
        self.W_bank = nn.Parameter(torch.empty(7, d, d))
        nn.init.orthogonal_(self.W_bank) # Initialize orthogonally
        
        # 4. Construct pre-compiled topologies
        self.register_buffer('M_normal', self._build_M_normal())
        self.register_buffer('M_crisis', self._build_M_crisis())
        
        # Create map from each active edge to its shared weight index in W_bank
        # We define a deterministic map to assign each possible edge (j, i) to one of the 7 matrices
        self.edge_to_w_idx = {}
        for i in range(28):
            for j in range(28):
                self.edge_to_w_idx[(j, i)] = (j + i) % 7
                
        # 5. Broyden Solver
        self.solver = BroydenSolver(max_iter=50, tol=1e-4, alpha=1.0)
        
        # 6. Mahalanobis distance monitor state (non-differentiable buffers)
        self.register_buffer('mu', torch.zeros(d_in))
        self.register_buffer('cov_inv', torch.eye(d_in))
        self.register_buffer('threshold', torch.tensor(10.0))
        self.register_buffer('warm_up_buffer', torch.zeros(warm_up_steps, d_in))
        self.register_buffer('warm_up_counter', torch.tensor(0, dtype=torch.long))
        self.register_buffer('is_calibrated', torch.tensor(False, dtype=torch.bool))
        
        # Active topology pointer (0 for normal, 1 for crisis)
        self.register_buffer('active_regime', torch.tensor(0, dtype=torch.long))

    def _build_M_normal(self) -> torch.Tensor:
        """
        Builds M_normal matching the paper's exact statistics:
        - Active Edges: 192
        - Max Path Length: 2 tiers (Sensory -> Reasoning, Sensory -> Constraint, no Reasoning -> Constraint)
        - Mean In-Deg (Reasoning): 9.4 (Total 94 edges from Sensory)
        - Mean In-Deg (Constraint): 8.1 (Total 57 edges from Sensory)
        - Intra-tier Sensory-to-Sensory connections: 41 edges (strictly lower-triangular, acyclic)
        """
        M = torch.zeros(28, 28)
        
        # 1. Sensory to Sensory (41 edges within 0..10, strictly j < i)
        sensory_pairs = [(j, i) for i in range(11) for j in range(i)]
        # Deterministically select 41 pairs
        selected_sensory = [sensory_pairs[k] for k in np.linspace(0, len(sensory_pairs)-1, 41, dtype=int)]
        for j, i in selected_sensory:
            M[i, j] = 1.0
            
        # 2. Sensory to Reasoning (94 edges from 0..10 to 11..20)
        reasoning_pairs = [(j, i) for i in range(11, 21) for j in range(11)]
        selected_reasoning = [reasoning_pairs[k] for k in np.linspace(0, len(reasoning_pairs)-1, 94, dtype=int)]
        for j, i in selected_reasoning:
            M[i, j] = 1.0
            
        # 3. Sensory to Constraint (57 edges from 0..10 to 21..27)
        constraint_pairs = [(j, i) for i in range(21, 28) for j in range(11)]
        selected_constraint = [constraint_pairs[k] for k in np.linspace(0, len(constraint_pairs)-1, 57, dtype=int)]
        for j, i in selected_constraint:
            M[i, j] = 1.0
            
        # Total active edges: 41 + 94 + 57 = 192. Perfect!
        return M

    def _build_M_crisis(self) -> torch.Tensor:
        """
        Builds M_crisis matching the paper's exact statistics:
        - Active Edges: 28
        - Max Path Length: 3 tiers (Sensory -> Reasoning -> Constraint)
        - Mean In-Deg (Reasoning): 1.0 (Strictly Tree, total 10 edges)
        - Mean In-Deg (Constraint): 2.5 (Total 18 edges from Reasoning to Constraint)
        """
        M = torch.zeros(28, 28)
        
        # 1. Sensory to Reasoning: 10 edges (each Reasoning agent receives from exactly 1 Sensory agent)
        for i in range(11, 21):
            j = (i - 11) % 11 # Map Reasoning agent to a Sensory agent
            M[i, j] = 1.0
            
        # 2. Reasoning to Constraint: 18 edges (from 11..20 to 21..27)
        # Constraint agents mean in-deg = 2.57 (some receive 2, some 3)
        constraint_in_degrees = [3, 3, 3, 3, 2, 2, 2] # Sum is 18
        curr_reasoner = 11
        for idx, i in enumerate(range(21, 28)):
            in_deg = constraint_in_degrees[idx]
            for _ in range(in_deg):
                M[i, curr_reasoner] = 1.0
                curr_reasoner = 11 + (curr_reasoner - 11 + 1) % 10
                
        # Total active edges: 10 + 18 = 28. Perfect!
        return M

    @torch.no_grad()
    def calibrate_mahalanobis(self):
        """
        Calibrates the Mahalanobis distance monitor using the collected warm-up observations.
        Estimates rolling mean, covariance inverse, and the 99.0th percentile threshold.
        """
        data = self.warm_up_buffer.cpu().numpy()
        
        # Compute mean
        self.mu.copy_(torch.tensor(np.mean(data, axis=0), device=self.mu.device))
        
        # Compute covariance
        cov = np.cov(data, rowvar=False)
        # Add regularization to prevent singularity
        cov += np.eye(self.d_in) * 1e-5
        
        # Inverse covariance
        cov_inv = np.linalg.inv(cov)
        self.cov_inv.copy_(torch.tensor(cov_inv, device=self.cov_inv.device))
        
        # Calculate distances for all warm-up points to find the threshold
        diffs = data - np.mean(data, axis=0)
        dists = np.sqrt(np.sum(np.dot(diffs, cov_inv) * diffs, axis=1))
        
        # Select 99.0th percentile threshold
        thresh = np.percentile(dists, 99.0)
        self.threshold.copy_(torch.tensor(thresh, device=self.threshold.device))
        self.is_calibrated.copy_(torch.tensor(True, dtype=torch.bool, device=self.threshold.device))

    def update_mahalanobis_monitor(self, x: torch.Tensor):
        """
        Monitors the incoming observation, computes Mahalanobis distance,
        and triggers Pre-Compiled Adjacency Switching (PCAS) if a shift is detected.
        """
        # If not calibrated, add to warm-up buffer
        if not self.is_calibrated:
            batch_size = x.size(0)
            # Just take the first sample of the batch for calibration simplification
            val = x[0].detach()
            counter = self.warm_up_counter.item()
            if counter < self.warm_up_steps:
                self.warm_up_buffer[counter] = val
                self.warm_up_counter.add_(1)
                
            if self.warm_up_counter.item() == self.warm_up_steps:
                self.calibrate_mahalanobis()
                
            # Stay in Normal regime during warm-up
            self.active_regime.copy_(torch.tensor(0, dtype=torch.long, device=x.device))
            return
            
        # Compute Mahalanobis distance for the incoming batch (average distance)
        with torch.no_grad():
            diff = x - self.mu.unsqueeze(0) # (B, d_in)
            # (B, d_in) @ (d_in, d_in) -> (B, d_in)
            temp = torch.matmul(diff, self.cov_inv)
            # Dot product along feature dimension
            dist = torch.sqrt(torch.sum(temp * diff, dim=-1)) # (B,)
            mean_dist = dist.mean().item()
            
            # PCAS trigger: swap adjacency matrix pointer if mean_dist > threshold
            if mean_dist > self.threshold.item():
                self.active_regime.copy_(torch.tensor(1, dtype=torch.long, device=x.device)) # Crisis
            else:
                self.active_regime.copy_(torch.tensor(0, dtype=torch.long, device=x.device)) # Normal

    def _compute_coupling(self, Z: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        Computes the continuous latent coupling tensors c_i for all agents.
        Z: concatenated states of shape (B, 28 * 256)
        M: active adjacency matrix of shape (28, 28)
        
        Returns coupling tensor of shape (B, 28, 256)
        """
        batch_size = Z.size(0)
        Z_reshaped = Z.view(batch_size, 28, self.d)
        
        # Get active edges: shape (NumEdges, 2) where each row is (i_target, j_source)
        targets, sources = torch.nonzero(M, as_tuple=True)
        
        if len(targets) == 0:
            return torch.zeros(batch_size, 28, self.d, device=Z.device)
            
        # Get the weight indices for these active edges
        w_indices = (targets + sources) % 7
        
        # Gather the weight matrices: shape (NumEdges, 256, 256)
        W_edges = self.W_bank[w_indices]
        
        # Gather the source states: shape (B, NumEdges, 256)
        z_sources = Z_reshaped[:, sources, :]
        
        # Compute W_ij * z_j: shape (B, NumEdges, 256)
        c_edges = torch.matmul(z_sources.unsqueeze(-2), W_edges.transpose(-1, -2)).squeeze(-2)
        
        # Scatter add to the coupling tensor: shape (B, 28, 256)
        C = torch.zeros(batch_size, 28, self.d, device=Z.device)
        targets_expanded = targets.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.d)
        C.scatter_add_(1, targets_expanded, c_edges)
        
        return C

    def forward_system_equations(self, Z: torch.Tensor, x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        Computes G(Z) for the joint fixed-point equation Z_next = G(Z).
        Z: concatenated states of shape (B, 28 * 256)
        x: raw environment observations of shape (B, d_in)
        M: active adjacency matrix of shape (28, 28)
        """
        batch_size = Z.size(0)
        Z_reshaped = Z.view(batch_size, 28, self.d)
        
        # 1. Compute continuous coupling tensors c_i for all agents
        C = self._compute_coupling(Z, M)
        
        # 2. Compute observation projections for sensory agents
        proj_X = torch.zeros(batch_size, 28, self.d, device=Z.device)
        for i in range(11):
            proj_X[:, i, :] = self.proj_layers[i](x)
            
        # 3. Compute next states for all 28 agents
        Z_next = torch.zeros(batch_size, 28, self.d, device=Z.device)
        for i in range(28):
            z_i = Z_reshaped[:, i, :]
            c_i = C[:, i, :]
            proj_xi = proj_X[:, i, :]
            
            # Forward pass through agent i
            Z_next[:, i, :] = self.agents[i](z_i, c_i, proj_xi)
            
        # Flatten to shape (B, 28 * 256)
        return Z_next.view(batch_size, 28 * self.d)

    def forward(self, x: torch.Tensor, z_prev: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Main forward pass of System 4 Swarm:
        1. Updates Mahalanobis monitor and dynamically swaps topology (PCAS)
        2. Solves for the joint fixed point Z* using Global Broyden Solver
        3. Extracts actions from the Constraint Projectors
        
        x: raw environment observations of shape (B, d_in)
        z_prev: optional initial guess for Z* of shape (B, 28 * 256)
        """
        batch_size = x.size(0)
        device = x.device
        
        # 1. Anomaly monitoring & PCAS topology switching
        self.update_mahalanobis_monitor(x)
        
        # Select active adjacency matrix
        if self.active_regime.item() == 1:
            M = self.M_crisis
            regime_name = "Crisis"
        else:
            M = self.M_normal
            regime_name = "Normal"
            
        # 2. Initialize fixed-point search
        if z_prev is None:
            z_init = torch.zeros(batch_size, 28 * self.d, device=device)
        else:
            z_init = z_prev.clone()
            
        # Solve for fixed point Z* = G(Z*)
        # We define G(Z) as a lambda function holding x and M fixed
        g_func = lambda Z: self.forward_system_equations(Z, x, M)
        
        Z_star, solver_info = self.solver.solve(g_func, z_init)
        
        # 3. Extract actions from the 7 Constraint Projectors (nodes 21-27)
        # We average the state vectors of the 7 constraint agents to project to output
        Z_star_reshaped = Z_star.view(batch_size, 28, self.d)
        constraint_states = Z_star_reshaped[:, 21:28, :] # shape (B, 7, 256)
        
        # Average across the 7 constraint agents and project to a small action dimension (say 4)
        # In actual environments, we will have a dedicated action head or let the controller baseline
        # map constraint states directly. Let's output the average representation as the joint latent action.
        latent_action = torch.mean(constraint_states, dim=1) # shape (B, 256)
        
        info = {
            "regime": regime_name,
            "regime_code": self.active_regime.item(),
            "solver": solver_info,
            "Z_star": Z_star
        }
        
        return latent_action, info
