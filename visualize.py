import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

from system4.swarm import System4Swarm
from system4.environments import FlashCrashEnv, QuadrotorTurbulenceEnv

def generate_visualization_plots(checkpoint_path: str = "system4_checkpoint.pt", save_dir: str = "plots"):
    print("=== Phase 5: Generating Trajectory and Performance Plots ===")
    
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found. Please run training first.")
        return
        
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    swarm = System4Swarm(d_in=64, d=256, gamma=0.98, warm_up_steps=100).to(device)
    swarm.load_state_dict(checkpoint['swarm_state_dict'])
    swarm.eval()
    
    # ----------------------------------------------------
    # 1. Run & Plot Task A (Flash Crash Trajectory)
    # ----------------------------------------------------
    print("Running Task A simulation for plotting...")
    env_a = FlashCrashEnv(steps=150)
    obs = env_a.reset()
    
    steps = []
    prices = []
    values = []
    drawdowns = []
    regimes = []
    
    done = False
    step_cnt = 0
    while not done:
        x_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            latent, info = swarm(x_tensor)
            action = torch.tanh(latent[:, :4]).squeeze(0).cpu().numpy()
            
        obs, reward, done, info_env = env_a.step(action)
        
        steps.append(step_cnt)
        prices.append(env_a.mid_price)
        values.append(env_a.cash + env_a.inventory * env_a.mid_price)
        drawdowns.append(info_env["drawdown"] * 100.0) # in %
        regimes.append(info["regime_code"])
        
        step_cnt += 1
        
    # Plot Task A
    sns.set_theme(style="darkgrid")
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Left axis: Portfolio Value
    color = 'tab:blue'
    ax1.set_xlabel('Time Steps (10 ms)', fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', color=color, fontweight='bold')
    line1 = ax1.plot(steps, values, color=color, linewidth=2.5, label='Portfolio Value')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Right axis: Mid Price
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Asset Mid Price ($)', color=color, fontweight='bold')
    line2 = ax2.plot(steps, prices, color=color, linestyle='--', linewidth=2.0, label='Asset Mid Price')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Highlight the Crisis regime (where regime_code == 1)
    regimes = np.array(regimes)
    crisis_idx = np.where(regimes == 1)[0]
    if len(crisis_idx) > 0:
        # Draw red vertical spans for crisis periods
        start = crisis_idx[0]
        for idx in range(1, len(crisis_idx)):
            if crisis_idx[idx] != crisis_idx[idx-1] + 1:
                ax1.axvspan(start, crisis_idx[idx-1], color='red', alpha=0.15)
                start = crisis_idx[idx]
        ax1.axvspan(start, crisis_idx[-1], color='red', alpha=0.15, label='Crisis Regime (PCAS Active)')
        
    plt.title('System 4: Portfolio Resilience During Financial Flash-Crash', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    plot_a_path = os.path.join(save_dir, 'flash_crash_trajectory.png')
    plt.savefig(plot_a_path, dpi=300)
    plt.close()
    print(f"Saved Flash Crash plot to {plot_a_path}")
    
    # ----------------------------------------------------
    # 2. Run & Plot Task B (Quadrotor Turbulence)
    # ----------------------------------------------------
    print("Running Task B simulation for plotting...")
    env_b = QuadrotorTurbulenceEnv(steps=150)
    obs = env_b.reset()
    
    steps = []
    rolls = []
    pitches = []
    winds = []
    regimes = []
    
    done = False
    step_cnt = 0
    while not done:
        x_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            latent, info = swarm(x_tensor)
            action = torch.tanh(latent[:, :2]).squeeze(0).cpu().numpy()
            
        obs, reward, done, info_env = env_b.step(action)
        
        steps.append(step_cnt)
        rolls.append(info_env["roll_deg"])
        pitches.append(info_env["pitch_deg"])
        winds.append(info_env["wind"])
        regimes.append(info["regime_code"])
        
        step_cnt += 1
        
    # Plot Task B
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    # Plot Roll/Pitch
    ax1.plot(steps, rolls, color='tab:blue', linewidth=2.0, label='Roll Deviation')
    ax1.plot(steps, pitches, color='tab:orange', linewidth=2.0, label='Pitch Deviation')
    ax1.axhline(15.0, color='red', linestyle=':', label='Survival Threshold (+/- 15°)')
    ax1.axhline(-15.0, color='red', linestyle=':')
    ax1.set_ylabel('Attitude Deviation (deg)', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_title('System 4: Quadrotor Stabilization Under Extreme Wind-Shear', fontsize=14, fontweight='bold')
    
    # Highlight crisis regime on top plot
    regimes = np.array(regimes)
    crisis_idx = np.where(regimes == 1)[0]
    if len(crisis_idx) > 0:
        start = crisis_idx[0]
        for idx in range(1, len(crisis_idx)):
            if crisis_idx[idx] != crisis_idx[idx-1] + 1:
                ax1.axvspan(start, crisis_idx[idx-1], color='red', alpha=0.15)
                ax2.axvspan(start, crisis_idx[idx-1], color='red', alpha=0.15)
                start = crisis_idx[idx]
        ax1.axvspan(start, crisis_idx[-1], color='red', alpha=0.15)
        ax2.axvspan(start, crisis_idx[-1], color='red', alpha=0.15)
        
    # Plot Wind Shear
    ax2.plot(steps, winds, color='tab:purple', linewidth=2.0, label='Wind Shear Magnitude')
    ax2.set_xlabel('Time Steps (10 ms)', fontweight='bold')
    ax2.set_ylabel('Wind Speed (m/s)', fontweight='bold')
    ax2.legend(loc='upper right')
    
    fig.tight_layout()
    plot_b_path = os.path.join(save_dir, 'quadrotor_attitude.png')
    plt.savefig(plot_b_path, dpi=300)
    plt.close()
    print(f"Saved Quadrotor plot to {plot_b_path}")
    
    # ----------------------------------------------------
    # 3. Plot Comparative Performance Chart (Hardcoded from Paper Table 2)
    # ----------------------------------------------------
    print("Generating comparative performance bar chart...")
    methods = ['PPO (Frozen)', 'PPO+Adapter', 'MPC', 'Sparse MoE', 'System 4 (Ours)']
    task_a_survival = [34.2, 68.1, 52.4, 67.5, 89.7]
    task_b_survival = [41.7, 72.4, 68.5, 70.1, 91.2]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, task_a_survival, width, label='Task A: Flash-Crash Survival %', color='royalblue')
    rects2 = ax.bar(x + width/2, task_b_survival, width, label='Task B: Quadrotor Survival %', color='mediumseagreen')
    
    ax.set_ylabel('Survival Rate (%)', fontweight='bold')
    ax.set_title('Comparative Benchmark Survival Rates', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 110)
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plot_c_path = os.path.join(save_dir, 'performance_comparison.png')
    plt.savefig(plot_c_path, dpi=300)
    plt.close()
    print(f"Saved comparison plot to {plot_c_path}")
    print("========================================================\n")

if __name__ == "__main__":
    generate_visualization_plots()
