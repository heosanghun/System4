import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import List, Tuple

from .swarm import System4Swarm
from .environments import StreamingClassificationEnv

def generate_synthetic_normal_data(classes: int = 10, d_in: int = 64, n_samples: int = 500) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates synthetic 'normal' dataset for pre-training.
    These represent uncorrupted, standard features centered around class manifolds in R^64.
    """
    np.random.seed(42)
    # 10 class centers
    class_centers = np.random.uniform(-1.0, 1.0, size=(classes, d_in))
    class_centers = class_centers / np.linalg.norm(class_centers, axis=1, keepdims=True)
    
    features = []
    labels = []
    
    for _ in range(n_samples):
        label = np.random.randint(0, classes)
        center = class_centers[label]
        # Low intra-class noise in normal regime
        noise = np.random.normal(0, 0.05, size=d_in)
        feat = center + noise
        feat = feat / np.linalg.norm(feat)
        
        features.append(feat)
        labels.append(label)
        
    return torch.from_numpy(np.array(features, dtype=np.float32)), torch.tensor(labels, dtype=torch.long)

def train_system4(epochs: int = 2, batch_size: int = 64, lr: float = 3e-4, checkpoint_path: str = "system4_checkpoint.pt"):
    """
    Offline training loop for System 4.
    Optimizes the agent weights, interaction weights, and projections.
    """
    print("=== Phase 4: Starting Offline Training for System 4 Swarm ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize Swarm with fast warm-up (100 steps)
    swarm = System4Swarm(d_in=64, d=256, gamma=0.98, warm_up_steps=100).to(device)
    
    # Classification head: projects average constraint state (256) to 10 classes
    class_head = nn.Linear(256, 10).to(device)
    
    # 2. Generate training data (256 samples is perfect for quick batching)
    features, labels = generate_synthetic_normal_data(classes=10, d_in=64, n_samples=256)
    
    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Setup optimizer (Adam with learning rate 3e-4)
    # Optimize all swarm parameters and classification head parameters
    optimizer = optim.Adam(
        list(swarm.parameters()) + list(class_head.parameters()),
        lr=lr
    )
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training loop
    for epoch in range(epochs):
        swarm.train()
        class_head.train()
        
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Warm up the Mahalanobis detector during training loader run
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: Solves equilibrium state
            latent_action, info = swarm(x)
            logits = class_head(latent_action)
            
            loss = criterion(logits, y)
            loss.backward()
            
            # Clip gradients to prevent instability
            torch.nn.utils.clip_grad_norm_(swarm.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item() * x.size(0)
            pred = torch.argmax(logits, dim=-1)
            correct += (pred == y).sum().item()
            total += x.size(0)
            
        acc = correct / total
        avg_loss = epoch_loss / total
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}%")
        
    # Check calibrated state
    print("Calibrating Mahalanobis detector...")
    # Feed warm-up observations to calibrate the monitor
    swarm.eval()
    for i in range(100):
        dummy_obs = features[i:i+1].to(device)
        swarm.update_mahalanobis_monitor(dummy_obs)
        
    print(f"Mahalanobis calibrated: {swarm.is_calibrated.item()}")
    print(f"Calibrated threshold: {swarm.threshold.item():.4f}")
    
    # Verify spectral properties of agents
    print("Checking agent spectral norms...")
    for idx, agent in enumerate(swarm.agents):
        norm = agent.estimated_lipschitz
        if idx % 7 == 0:
            print(f"Agent A_{idx:02d} Lipschitz bound: {norm:.4f}")
            
    # Save checkpoint
    torch.save({
        'swarm_state_dict': swarm.state_dict(),
        'class_head_state_dict': class_head.state_dict(),
        'class_centers': StreamingClassificationEnv().class_centers
    }, checkpoint_path)
    
    print(f"Checkpoint successfully saved to {checkpoint_path}")
    print("========================================================\n")
    return swarm, class_head

if __name__ == "__main__":
    train_system4(epochs=2)
