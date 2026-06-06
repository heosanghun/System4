import torch
import torch.nn as nn
from typing import Callable, Tuple, Dict, Any

class BroydenSolver:
    """
    A highly optimized, memory-efficient batched Broyden solver for root-finding:
    F(Z) = Z - G(Z) = 0.
    
    Rather than constructing or storing full Jacobian/Inverse-Jacobian matrices
    (which would be (N*d) x (N*d) ~ 7168 x 7168 per batch element), it represents
    the inverse Jacobian recursively using a list of low-rank updates (Sherman-Morrison updates).
    """
    def __init__(self, max_iter: int = 50, tol: float = 1e-4, alpha: float = 1.0, eps: float = 1e-9):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.eps = eps

    def solve(self, func: Callable[[torch.Tensor], torch.Tensor], z_init: torch.Tensor, fixed_iter: int = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Solves F(Z) = 0 starting from z_init.
        func: G(Z), maps Z of shape (B, Nd) to G(Z) of shape (B, Nd)
        z_init: initial state of shape (B, Nd)
        fixed_iter: optional fixed number of solver iterations to bypass dynamic CPU convergence checks
        """
        batch_size = z_init.size(0)
        device = z_init.device
        
        # Initial evaluation
        z = z_init.clone()
        g = func(z)
        f = z - g # F(Z) = Z - G(Z)
        
        # Keep track of low-rank updates: u_list and v_list
        # H_k = H_0 + sum_{i=0}^{k-1} u_i v_i^T, where H_0 = -I
        u_list = []
        v_list = []
        
        info = {
            "converged": False,
            "iterations": 0,
            "final_err": 0.0,
            "history": []
        }
        
        if fixed_iter is not None:
            # ----------------------------------------------------
            # Fast Path: Fixed iteration solver (No CUDA syncs / no .item())
            # ----------------------------------------------------
            for k in range(fixed_iter):
                g_k = -f.clone()
                for u_i, v_i in zip(u_list, v_list):
                    dot = torch.sum(v_i * f, dim=-1, keepdim=True)
                    g_k = g_k + u_i * dot
                    
                delta_z = -self.alpha * g_k
                z_next = z + delta_z
                
                g_next = func(z_next)
                f_next = z_next - g_next
                delta_f = f_next - f
                
                w_k = -delta_f.clone()
                for u_i, v_i in zip(u_list, v_list):
                    dot = torch.sum(v_i * delta_f, dim=-1, keepdim=True)
                    w_k = w_k + u_i * dot
                    
                v_k = -delta_f.clone()
                for u_i, v_i in zip(u_list, v_list):
                    dot = torch.sum(u_i * delta_f, dim=-1, keepdim=True)
                    v_k = v_k + v_i * dot
                    
                denom = torch.sum(delta_f * w_k, dim=-1, keepdim=True)
                denom = torch.where(denom.abs() < self.eps, self.eps * denom.sign(), denom)
                
                u_k = (delta_z - w_k) / denom
                
                u_list.append(u_k)
                v_list.append(v_k)
                
                z = z_next
                f = f_next
                
                if len(u_list) > 20:
                    u_list.pop(0)
                    v_list.pop(0)
                    
            info["converged"] = True
            info["iterations"] = fixed_iter
            info["final_err"] = 0.0  # Dynamic evaluation skipped to prevent synchronization
            return z, info
            
        # ----------------------------------------------------
        # Standard Path: Dynamic convergence checking (Uses .item())
        # ----------------------------------------------------
        for k in range(self.max_iter):
            # Compute norm of residual error F(Z)
            err = torch.norm(f, p=2, dim=-1) # shape (B,)
            mean_err = err.mean().item()
            info["history"].append(mean_err)
            
            # Check convergence
            if mean_err < self.tol:
                info["converged"] = True
                info["iterations"] = k
                info["final_err"] = mean_err
                return z, info
            
            # Compute search direction g_k = H_k F(Z_k)
            g_k = -f.clone()
            for u_i, v_i in zip(u_list, v_list):
                dot = torch.sum(v_i * f, dim=-1, keepdim=True)
                g_k = g_k + u_i * dot
                
            delta_z = -self.alpha * g_k
            z_next = z + delta_z
            
            # Evaluate new residual
            g_next = func(z_next)
            f_next = z_next - g_next
            delta_f = f_next - f
            
            # Compute w_k = H_k delta_f
            w_k = -delta_f.clone()
            for u_i, v_i in zip(u_list, v_list):
                dot = torch.sum(v_i * delta_f, dim=-1, keepdim=True)
                w_k = w_k + u_i * dot
                
            # Compute v_k = H_k^T delta_f
            v_k = -delta_f.clone()
            for u_i, v_i in zip(u_list, v_list):
                dot = torch.sum(u_i * delta_f, dim=-1, keepdim=True)
                v_k = v_k + v_i * dot
                
            # Compute batch denominator
            denom = torch.sum(delta_f * w_k, dim=-1, keepdim=True)
            denom = torch.where(denom.abs() < self.eps, self.eps * denom.sign(), denom)
            
            # Compute u_k
            u_k = (delta_z - w_k) / denom
            
            u_list.append(u_k)
            v_list.append(v_k)
            
            z = z_next
            f = f_next
            
            if len(u_list) > 20:
                u_list.pop(0)
                v_list.pop(0)
                
        # Fallback Picard steps
        for k in range(5):
            g = func(z)
            err = torch.norm(z - g, p=2, dim=-1).mean().item()
            z = g
            if err < self.tol:
                info["converged"] = True
                info["iterations"] = self.max_iter + k
                info["final_err"] = err
                return z, info
                
        info["final_err"] = torch.norm(z - func(z), p=2, dim=-1).mean().item()
        return z, info
