import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import List, Tuple, Dict, Any

from .swarm import System4Swarm
from .environments import FlashCrashEnv, QuadrotorTurbulenceEnv, StreamingClassificationEnv
from .baselines import MPCController

def collect_mpc_demonstrations(task: str, n_episodes: int = 10, steps_per_ep: int = 90) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collects expert state-action trajectories by running the MPC baseline controller
    under normal conditions (no crash, low wind).
    """
    print(f"Collecting MPC demonstrations for {task}...")
    obs_list = []
    act_list = []
    
    mpc = MPCController(action_dim=4 if task == "flash_crash" else 2, horizon=10)
    
    for ep in range(n_episodes):
        if task == "flash_crash":
            # Normal conditions: steps < 100, no crash
            env = FlashCrashEnv(steps=steps_per_ep)
        else:
            # Normal conditions: steps < 100, low wind
            env = QuadrotorTurbulenceEnv(steps=steps_per_ep)
            
        obs = env.reset()
        done = False
        
        while not done:
            action = mpc.get_action(obs, task)
            if task == "turbulence":
                action = action[:2] # Quadrotor roll/pitch only
                
            obs_list.append(np.copy(obs))
            act_list.append(np.copy(action))
            
            obs, reward, done, _ = env.step(action)
            
    obs_arr = np.array(obs_list, dtype=np.float32)
    act_arr = np.array(act_list, dtype=np.float32)
    
    return torch.from_numpy(obs_arr), torch.from_numpy(act_arr)

def generate_synthetic_normal_data(classes: int = 10, d_in: int = 64, n_samples: int = 500) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates synthetic 'normal' dataset for pre-training classification.
    """
    np.random.seed(42)
    class_centers = np.random.uniform(-1.0, 1.0, size=(classes, d_in))
    class_centers = class_centers / np.linalg.norm(class_centers, axis=1, keepdims=True)
    
    features = []
    labels = []
    
    for _ in range(n_samples):
        label = np.random.randint(0, classes)
        center = class_centers[label]
        noise = np.random.normal(0, 0.05, size=d_in)
        feat = center + noise
        feat = feat / np.linalg.norm(feat)
        
        features.append(feat)
        labels.append(label)
        
    return torch.from_numpy(np.array(features, dtype=np.float32)), torch.tensor(labels, dtype=torch.long)

def train_system4(epochs: int = 3, batch_size: int = 64, lr: float = 3e-4, checkpoint_path: str = "system4_checkpoint.pt"):
    print("=== Phase 7: Starting Multi-Task Behavioral Cloning for System 4 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Collect MPC expert data
    x_a, y_a = collect_mpc_demonstrations("flash_crash", n_episodes=5, steps_per_ep=90)
    x_b, y_b = collect_mpc_demonstrations("turbulence", n_episodes=5, steps_per_ep=90)
    x_c, y_c = generate_synthetic_normal_data(classes=10, d_in=64, n_samples=500)
    
    # Create DataLoader for each task
    loader_a = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_a, y_a), batch_size=batch_size, shuffle=True)
    loader_b = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_b, y_b), batch_size=batch_size, shuffle=True)
    loader_c = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_c, y_c), batch_size=batch_size, shuffle=True)
    
    # 2. Initialize Swarm
    swarm = System4Swarm(d_in=64, d=256, gamma=0.98, warm_up_steps=100).to(device)
    class_head = nn.Linear(256, 10).to(device)
    
    # 3. Setup optimizer
    optimizer = optim.Adam(
        list(swarm.parameters()) + list(class_head.parameters()),
        lr=lr
    )
    
    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()
    
    # 4. Training loop
    for epoch in range(epochs):
        swarm.train()
        class_head.train()
        
        loss_a_total = 0.0
        loss_b_total = 0.0
        loss_c_total = 0.0
        
        # We zip the loaders (they might have different lengths, so we zip up to the shortest)
        for (batch_a, batch_b, batch_c) in zip(loader_a, loader_b, loader_c):
            optimizer.zero_grad()
            
            # Unpack batches
            xa, ya = batch_a[0].to(device), batch_a[1].to(device)
            xb, yb = batch_b[0].to(device), batch_b[1].to(device)
            xc, yc = batch_c[0].to(device), batch_c[1].to(device)
            
            # --- Task A: Flash Crash (Action shape B, 4) ---
            latent_a, _ = swarm(xa)
            pred_a = torch.tanh(latent_a[:, :4])
            loss_a = mse_criterion(pred_a, ya)
            
            # --- Task B: Turbulence (Action shape B, 2) ---
            latent_b, _ = swarm(xb)
            pred_b = torch.tanh(latent_b[:, :2])
            loss_b = mse_criterion(pred_b, yb)
            
            # --- Task C: Classification (CrossEntropy) ---
            latent_c, _ = swarm(xc)
            logits_c = class_head(latent_c)
            loss_c = ce_criterion(logits_c, yc)
            
            # Joint Loss
            loss = loss_a + loss_b + loss_c
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(swarm.parameters(), max_norm=1.0)
            optimizer.step()
            
            loss_a_total += loss_a.item()
            loss_b_total += loss_b.item()
            loss_c_total += loss_c.item()
            
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Task A MSE: {loss_a_total:.4f} | Task B MSE: {loss_b_total:.4f} | Task C CE: {loss_c_total:.4f}")
        
    # Calibrate Mahalanobis detector
    print("Calibrating Mahalanobis detector...")
    swarm.eval()
    for i in range(100):
        dummy_obs = x_c[i:i+1].to(device)
        swarm.update_mahalanobis_monitor(dummy_obs)
        
    print(f"Mahalanobis calibrated: {swarm.is_calibrated.item()}")
    print(f"Calibrated threshold: {swarm.threshold.item():.4f}")
    
    # Save checkpoint
    torch.save({
        'swarm_state_dict': swarm.state_dict(),
        'class_head_state_dict': class_head.state_dict(),
        'class_centers': StreamingClassificationEnv().class_centers
    }, checkpoint_path)
    
    print(f"Multi-Task Checkpoint saved to {checkpoint_path}")
    print("========================================================\n")
    return swarm, class_head

if __name__ == "__main__":
    train_system4()
