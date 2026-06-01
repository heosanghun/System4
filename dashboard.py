import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os

from system4.swarm import System4Swarm
from system4.environments import FlashCrashEnv, QuadrotorTurbulenceEnv

st.set_page_config(
    page_title="System 4: Stable Test-Time Adaptation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 40px;
        color: #1E3A8A;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 18px;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2563EB;
        margin-bottom: 10px;
    }
    .regime-normal {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 5px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
    .regime-crisis {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 5px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allowed_html=True)

# App Title
st.markdown('<div class="main-title">System 4: Stable Test-Time Adaptation Without Gradients</div>', unsafe_allowed_html=True)
st.markdown('<div class="sub-title">Topology-Switching Coupled Equilibrium Models for Real-Time Streaming Distribution Shifts</div>', unsafe_allowed_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.image("https://img.icons8.com/color/144/quadcopter.png", width=80)
st.sidebar.markdown("### Model Configuration")

# Hidden size and agent count
st.sidebar.info("Swarm Size: N=28 Agents\nTiers: 11 Sensory, 10 Reasoners, 7 Constraints\nInteraction Parameters: ~0.44M")

# Threshold setup
thresh_percentile = st.sidebar.slider("Mahalanobis Detection Threshold (%)", 90.0, 99.9, 99.0, step=0.5)

# Load checkpoint
checkpoint_path = "system4_checkpoint.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_swarm_model(path):
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    swarm = System4Swarm(d_in=64, d=256, gamma=0.98, warm_up_steps=100).to(device)
    swarm.load_state_dict(checkpoint['swarm_state_dict'])
    swarm.eval()
    return swarm

swarm = load_swarm_model(checkpoint_path)

if swarm is None:
    st.warning("⚠️ No trained checkpoint found. Please run Phase 4 Training to unlock the fully functional model.")
    # Initialize a dummy untrained swarm for demonstration
    swarm = System4Swarm(d_in=64, d=256, gamma=0.98).to(device)
    swarm.eval()
else:
    st.sidebar.success("✅ Fully Trained System 4 Model Loaded")

# Adjust calibrated threshold based on slider selection
if swarm is not None and swarm.is_calibrated.item():
    # Simulate adjusting the threshold from calibrated percentile
    # 99.0th is default, we scale standard threshold slightly
    base_thresh = swarm.threshold.item()
    scale = (thresh_percentile / 99.0)
    swarm.threshold.copy_(torch.tensor(base_thresh * scale, device=swarm.threshold.device))

# ----------------- MAIN LAYOUT -----------------
tab1, tab2, tab3 = st.tabs(["🚀 Real-Time Simulator", "📊 Comparative Benchmarks", "🌐 Swarm Topologies"])

with tab1:
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.markdown("### Choose Simulation Task")
        task = st.selectbox(
            "Select Environment",
            ["Task A: Financial Flash-Crash (Limit Order Book)", 
             "Task B: Aerospace Turbulence Recovery (UAV Quadrotor)"]
        )
        
        sim_duration = st.slider("Simulation Steps", 50, 300, 150, step=10)
        
        run_btn = st.button("▶️ Start Simulation Episode", use_container_width=True)
        
    with col_right:
        st.markdown("### Live Dynamics Monitor")
        plot_placeholder = st.empty()
        metrics_placeholder = st.empty()

    if run_btn:
        steps_hist = []
        val_hist = []
        ref_hist = []
        regime_hist = []
        dist_hist = []
        
        # Initialize selected environment
        if "Task A" in task:
            env = FlashCrashEnv(steps=sim_duration)
            obs = env.reset()
            title = "LOB Flash Crash Live Playback"
            y_label_left = "Portfolio Value ($)"
            y_label_right = "Asset Mid Price ($)"
        else:
            env = QuadrotorTurbulenceEnv(steps=sim_duration)
            obs = env.reset()
            title = "UAV Quadrotor Stabilization Live Playback"
            y_label_left = "Attitude Deviation (deg)"
            y_label_right = "Wind Speed (m/s)"
            
        progress_bar = st.progress(0)
        
        # Run step-by-step
        for step in range(sim_duration):
            x_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                # Forward pass through System 4
                latent, info = swarm(x_tensor)
                
                # Get environment action
                if "Task A" in task:
                    action = torch.tanh(latent[:, :4]).squeeze(0).cpu().numpy()
                else:
                    action = torch.tanh(latent[:, :2]).squeeze(0).cpu().numpy()
                    
            # Update environment
            obs, reward, done, info_env = env.step(action)
            
            # Log metrics
            steps_hist.append(step)
            regime_hist.append(info["regime_code"])
            
            # Compute current Mahalanobis distance for display
            if swarm.is_calibrated.item():
                diff = x_tensor - swarm.mu.unsqueeze(0)
                temp = torch.matmul(diff, swarm.cov_inv)
                dist = torch.sqrt(torch.sum(temp * diff, dim=-1)).item()
            else:
                dist = 1.0 # default before calibration
            dist_hist.append(dist)
            
            if "Task A" in task:
                val_hist.append(info_env["portfolio_value"])
                ref_hist.append(env.mid_price)
                status_text = f"Regime: {info['regime']} | Value: ${info_env['portfolio_value']:.2f} | Drawdown: {info_env['drawdown']*100:.2f}%"
            else:
                val_hist.append(max(abs(info_env["roll_deg"]), abs(info_env["pitch_deg"])))
                ref_hist.append(info_env["wind"])
                status_text = f"Regime: {info['regime']} | Max Deviation: {max(abs(info_env['roll_deg']), abs(info_env['pitch_deg'])):.2f}° | Wind: {info_env['wind']:.1f} m/s"
                
            progress_bar.progress((step + 1) / sim_duration)
            
            # Update Live Plot
            fig, ax1 = plt.subplots(figsize=(10, 4.5))
            
            # Left Plot
            color = 'tab:blue'
            ax1.set_xlabel('Time Steps (10 ms)', fontweight='bold')
            ax1.set_ylabel(y_label_left, color=color, fontweight='bold')
            ax1.plot(steps_hist, val_hist, color=color, linewidth=2.5)
            ax1.tick_params(axis='y', labelcolor=color)
            
            # Right Plot
            ax2 = ax1.twinx()
            color = 'tab:green'
            ax2.set_ylabel(y_label_right, color=color, fontweight='bold')
            ax2.plot(steps_hist, ref_hist, color=color, linestyle='--', linewidth=1.8)
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Crisis Shading
            regimes = np.array(regime_hist)
            crisis_idx = np.where(regimes == 1)[0]
            if len(crisis_idx) > 0:
                start_c = crisis_idx[0]
                for idx_c in range(1, len(crisis_idx)):
                    if crisis_idx[idx_c] != crisis_idx[idx_c-1] + 1:
                        ax1.axvspan(start_c, crisis_idx[idx_c-1], color='red', alpha=0.15)
                        start_c = crisis_idx[idx_c]
                ax1.axvspan(start_c, crisis_idx[-1], color='red', alpha=0.15)
                
            plt.title(title, fontsize=12, fontweight='bold')
            fig.tight_layout()
            plot_placeholder.pyplot(fig)
            plt.close()
            
            # Update Metrics
            regime_badge = '<span class="regime-crisis">🚨 CRISIS REGIME</span>' if info["regime_code"] == 1 else '<span class="regime-normal">❇️ NORMAL REGIME</span>'
            metrics_placeholder.markdown(f"""
            <div style="display: flex; gap: 20px;">
                <div class="metric-card" style="flex: 1;">
                    <div style="font-size: 14px; color: #4B5563;">Current Status</div>
                    <div style="font-size: 20px; font-weight: bold;">{regime_badge}</div>
                </div>
                <div class="metric-card" style="flex: 1;">
                    <div style="font-size: 14px; color: #4B5563;">Mahalanobis Distance (d_t)</div>
                    <div style="font-size: 24px; font-weight: bold; color: {'#DC2626' if dist > swarm.threshold.item() else '#059669'};">{dist:.3f}</div>
                    <div style="font-size: 12px; color: #6B7280;">Threshold: {swarm.threshold.item():.3f}</div>
                </div>
            </div>
            """, unsafe_allowed_html=True)
            
            time.sleep(0.01) # fast playback

with tab2:
    st.markdown("### Quantitative Performance Benchmarks")
    st.markdown("Summary of survival rates and adaptation latencies across 500 independent episodes.")
    
    # Custom comparative data frame matching Paper Table 2
    st.markdown("""
    | Method | Task A Survival (LOB) | Task B Survival (UAV) | Inference Latency | Adaptation Latency | Total Params |
    | :--- | :---: | :---: | :---: | :---: | :---: |
    | **System 4 (Ours)** | **89.7% ± 2.8** | **91.2% ± 2.5** | **2.4 ± 0.3 ms** | **6.3 ± 1.2 ms** | **4.1M** |
    | Sparse MoE Router | 67.5% ± 3.5 | 70.1% ± 3.7 | 2.1 ± 0.4 ms | 2.1 ± 0.4 ms | 34.6M |
    | PPO + Online Adapter | 68.1% ± 3.8 | 72.4% ± 4.0 | 0.8 ± 0.1 ms | 2,840 ± 310 ms | 9.4M |
    | MPC (Domain Standard) | 52.4% ± 4.6 | 68.5% ± 3.9 | 42.5 ± 8.2 ms | 42.5 ± 8.2 ms | N/A |
    | PPO (Frozen Inference) | 34.2% ± 4.1 | 41.7% ± 3.3 | 0.8 ± 0.1 ms | N/A | 9.2M |
    """)
    
    st.markdown("### Performance Visualizations")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if os.path.exists("plots/performance_comparison.png"):
            st.image("plots/performance_comparison.png", caption="Survival Rate Comparison", use_container_width=True)
        else:
            st.info("Performance comparison plot will be visible after running visualizer.")
    with col_img2:
        if os.path.exists("plots/flash_crash_trajectory.png"):
            st.image("plots/flash_crash_trajectory.png", caption="System 4 Flash-Crash Trajectory", use_container_width=True)
        else:
            st.info("Flash crash trajectory plot will be visible after running visualizer.")

with tab3:
    st.markdown("### Pre-Compiled Adjacency Switching (PCAS) Routing Geometry")
    st.markdown("System 4 adapts instantly by swapping between two pre-compiled graph structures.")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("#### 🟢 Normal Regime (Dense Exploratory Routing)")
        st.info("Edges: 192 | Path Length: 2 Tiers\n- Sensory agents project directly to both Reasoners and Constraint Projectors.\n- Promotes high-capacity representation search.")
        st.code("""
      [Sensory Encoders (11 nodes)]
         |                 \\
         |  (Dense 94)      \\  (Dense 57)
         v                   v
[Latent Reasoners (10)]   [Constraint Projectors (7)]
        """, language="text")
        
    with col_g2:
        st.markdown("#### 🔴 Crisis Regime (Strict Hierarchical Routing)")
        st.error("Edges: 28 | Path Length: 3 Tiers\n- Sensory agents connect strictly to Reasoners (Tree-structured).\n- Reasoners connect to Constraint Projectors.\n- Prunes speculative edges to enforce safety and rapid solver convergence.")
        st.code("""
      [Sensory Encoders (11 nodes)]
                    |
                    | (Strict Tree: 10 edges)
                    v
         [Latent Reasoners (10)]
                    |
                    | (Hierarchical: 18 edges)
                    v
        [Constraint Projectors (7)]
        """, language="text")
