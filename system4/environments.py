import numpy as np
import torch
from typing import Tuple, Dict, Any

class FlashCrashEnv:
    """
    High-fidelity Limit Order Book (LOB) Simulator for Task A (Financial Flash-Crash).
    Simulates price dynamics, bid/ask depth at 5 levels, order flows, and sudden
    adversarial liquidity-withdrawal events (mimicking the May 2010 Flash Crash).
    
    Survival is defined as maintaining portfolio drawdown strictly below 5% over
    a 500 ms (50 steps of 10ms) episode.
    """
    def __init__(self, steps: int = 500, d_in: int = 64):
        self.steps = steps
        self.d_in = d_in
        self.reset()

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.mid_price = 100.0
        self.cash = 10000.0
        self.inventory = 100  # Start with 100 shares
        self.initial_portfolio_value = self.cash + self.inventory * self.mid_price
        self.max_portfolio_value = self.initial_portfolio_value
        
        # Volatility and book dynamics
        self.base_volatility = 0.1
        self.volatility = self.base_volatility
        
        # 5 levels of Bid/Ask depth
        self.bid_prices = np.array([self.mid_price - 0.1 * i for i in range(1, 6)])
        self.bid_sizes = np.random.randint(100, 500, size=5).astype(float)
        self.ask_prices = np.array([self.mid_price + 0.1 * i for i in range(1, 6)])
        self.ask_sizes = np.random.randint(100, 500, size=5).astype(float)
        
        # History for observations
        self.price_history = [self.mid_price] * 10
        self.drawdown_history = [0.0] * 10
        self.vol_history = [self.volatility] * 10
        
        self.crash_triggered = False
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        # Build a 64-dimensional observation vector containing:
        # - Mid price (1)
        # - Portfolio metrics: cash, inventory, value, drawdown, max_value (5)
        # - Bid prices & sizes (10)
        # - Ask prices & sizes (10)
        # - Volatility & Spread (2)
        # - History vectors (30)
        # - Padding (6)
        obs = np.zeros(self.d_in, dtype=np.float32)
        obs[0] = self.mid_price / 100.0
        obs[1] = self.cash / 10000.0
        obs[2] = self.inventory / 100.0
        obs[3] = (self.cash + self.inventory * self.mid_price) / self.initial_portfolio_value
        obs[4] = self._get_drawdown()
        obs[5] = self.max_portfolio_value / self.initial_portfolio_value
        
        obs[6:11] = self.bid_prices / 100.0
        obs[11:16] = self.bid_sizes / 500.0
        obs[16:21] = self.ask_prices / 100.0
        obs[21:26] = self.ask_sizes / 500.0
        obs[26] = self.volatility
        obs[27] = (self.ask_prices[0] - self.bid_prices[0]) / 0.2
        
        # Add history
        obs[28:38] = np.array(self.price_history[-10:]) / 100.0
        obs[38:48] = np.array(self.drawdown_history[-10:])
        obs[48:58] = np.array(self.vol_history[-10:])
        
        # Padding with noise
        obs[58:] = np.random.normal(0, 0.01, size=6)
        return obs

    def _get_drawdown(self) -> float:
        val = self.cash + self.inventory * self.mid_price
        if val > self.max_portfolio_value:
            self.max_portfolio_value = val
        drawdown = (self.max_portfolio_value - val) / self.max_portfolio_value
        return float(drawdown)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        action: 4 continuous values in [-1, 1] representing:
        - Place limit buy size (0 to 50 shares)
        - Place limit sell size (0 to 50 shares)
        - Place market buy size (0 to 50 shares)
        - Place market sell size (0 to 50 shares)
        """
        self.current_step += 1
        
        # Trigger crash in the middle of the episode (e.g. step 150)
        # Or if observations start shifting
        if self.current_step >= 100:
            self.crash_triggered = True
            
        # 1. Update Market Book Dynamics
        if self.crash_triggered:
            # Crash: High volatility, massive bid liquidity withdrawal, huge price drop
            self.volatility = self.base_volatility * 15.0
            # Withdraw 95% of bid sizes
            self.bid_sizes = self.bid_sizes * 0.05 + np.random.randint(1, 10, size=5)
            # Price drops dramatically
            price_drop = np.random.normal(-1.5, 0.5)
            self.mid_price = max(10.0, self.mid_price + price_drop)
        else:
            # Normal: Low volatility, healthy depths
            self.volatility = self.base_volatility
            self.bid_sizes = self.bid_sizes * 0.9 + np.random.randint(50, 150, size=5)
            self.ask_sizes = self.ask_sizes * 0.9 + np.random.randint(50, 150, size=5)
            price_change = np.random.normal(0, 0.1)
            self.mid_price += price_change
            
        # Re-index bids and asks
        self.bid_prices = np.array([self.mid_price - 0.1 * i for i in range(1, 6)])
        self.ask_prices = np.array([self.mid_price + 0.1 * i for i in range(1, 6)])
        
        # 2. Process Agent Actions
        # Scale actions to [0, 50]
        limit_buy_size = max(0.0, (action[0] + 1) * 25.0)
        limit_sell_size = max(0.0, (action[1] + 1) * 25.0)
        market_buy_size = max(0.0, (action[2] + 1) * 25.0)
        market_sell_size = max(0.0, (action[3] + 1) * 25.0)
        
        # Execute Market Sell (sells inventory to bids)
        if market_sell_size > 0 and self.inventory > 0:
            sold_shares = min(market_sell_size, self.inventory)
            # Match against bids
            price = self.bid_prices[0]
            self.cash += sold_shares * price
            self.inventory -= sold_shares
            
        # Execute Market Buy (buys inventory from asks)
        if market_buy_size > 0:
            price = self.ask_prices[0]
            max_buy = self.cash / price
            bought_shares = min(market_buy_size, max_buy)
            self.cash -= bought_shares * price
            self.inventory += bought_shares
            
        # Process limit orders (simplified execution)
        if limit_buy_size > 0:
            # If price drops below bid, execute
            if self.mid_price <= self.bid_prices[1]:
                bought = min(limit_buy_size, self.cash / self.bid_prices[1])
                self.cash -= bought * self.bid_prices[1]
                self.inventory += bought
                
        if limit_sell_size > 0 and self.inventory > 0:
            # If price rises above ask, execute
            if self.mid_price >= self.ask_prices[1]:
                sold = min(limit_sell_size, self.inventory)
                self.cash += sold * self.ask_prices[1]
                self.inventory -= sold
                
        # 3. Calculate portfolio and drawdown
        portfolio_value = self.cash + self.inventory * self.mid_price
        drawdown = self._get_drawdown()
        
        # Update histories
        self.price_history.append(self.mid_price)
        self.drawdown_history.append(drawdown)
        self.vol_history.append(self.volatility)
        
        # 4. Check survival (drawdown < 5%)
        # Let's say if drawdown exceeds 5%, the episode is a failure
        failed = drawdown >= 0.05
        done = self.current_step >= self.steps or failed
        
        # Reward is based on maintaining value and minimizing drawdown
        reward = -drawdown
        if failed:
            reward -= 10.0 # Heavy penalty
            
        info = {
            "portfolio_value": portfolio_value,
            "drawdown": drawdown,
            "failed": failed,
            "crash": self.crash_triggered
        }
        
        return self._get_observation(), reward, done, info


class QuadrotorTurbulenceEnv:
    """
    High-fidelity 2D UAV Quadrotor physical simulator for Task B (Aerospace Turbulence Recovery).
    Simulates roll (phi), pitch (theta), and angular velocities subject to wind-shear
    disturbances modeled using a heavy-tailed Student-t distribution (nu=3, peak 25m/s).
    
    Survival requires maintaining roll and pitch deviations within +/- 15 degrees
    throughout a 500-step episode.
    """
    def __init__(self, steps: int = 500, d_in: int = 64):
        self.steps = steps
        self.d_in = d_in
        # Physics constants
        self.Ix = 0.01  # Moment of inertia
        self.Iy = 0.01
        self.L = 0.25   # Arm length
        self.dt = 0.01  # 10ms control step
        self.reset()

    def reset(self) -> np.ndarray:
        self.current_step = 0
        # State: [roll, pitch, roll_rate, pitch_rate]
        self.state = np.zeros(4, dtype=float)
        
        # History
        self.state_history = [np.copy(self.state)] * 10
        self.wind_history = [0.0] * 10
        
        self.failed = False
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        # Build 64-dimensional observation vector
        # - Current state (4)
        # - History vectors (40)
        # - Wind history (10)
        # - Padding (10)
        obs = np.zeros(self.d_in, dtype=np.float32)
        obs[0:4] = self.state
        
        # Fill state history
        flat_hist = np.array(self.state_history[-10:]).flatten() # (40,)
        obs[4:44] = flat_hist
        
        # Wind history
        obs[44:54] = np.array(self.wind_history[-10:])
        
        # Noise padding
        obs[54:] = np.random.normal(0, 0.01, size=10)
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        action: 2 continuous values in [-1, 1] representing differential motor thrusts:
        - Roll control thrust
        - Pitch control thrust
        """
        self.current_step += 1
        
        # 1. Wind Shear Disturbance: Heavy-tailed Student-t (nu=3) with peak 25m/s
        # Scale to match the moments of inertia
        wind_scale = 5.0 if self.current_step >= 100 else 0.1 # Wind storm hits after step 100
        # Student-t random sample (df=3)
        wind_roll = np.random.standard_t(df=3) * wind_scale
        wind_pitch = np.random.standard_t(df=3) * wind_scale
        
        # 2. Physics Equations of Motion
        roll, pitch, p, q = self.state
        
        # Motor thrust controls
        u_roll = action[0] * 2.0
        u_pitch = action[1] * 2.0
        
        # Angular accelerations
        p_dot = (u_roll * self.L + wind_roll) / self.Ix
        q_dot = (u_pitch * self.L + wind_pitch) / self.Iy
        
        # Euler integration
        p_next = p + p_dot * self.dt
        q_next = q + q_dot * self.dt
        
        roll_next = roll + p_next * self.dt
        pitch_next = pitch + q_next * self.dt
        
        # Clip state for safety
        self.state = np.array([
            np.clip(roll_next, -np.pi, np.pi),
            np.clip(pitch_next, -np.pi, np.pi),
            np.clip(p_next, -20.0, 20.0),
            np.clip(q_next, -20.0, 20.0)
        ])
        
        # Check survival limits (+/- 15 degrees)
        # 15 degrees = 15 * pi / 180 = 0.2618 radians
        limit = 15.0 * np.pi / 180.0
        roll_deg = abs(self.state[0]) * 180.0 / np.pi
        pitch_deg = abs(self.state[1]) * 180.0 / np.pi
        
        if abs(self.state[0]) > limit or abs(self.state[1]) > limit:
            self.failed = True
            
        done = self.current_step >= self.steps or self.failed
        
        # Reward is based on stabilization
        reward = -(self.state[0]**2 + self.state[1]**2 + 0.1 * p**2 + 0.1 * q**2)
        if self.failed:
            reward -= 100.0 # Massive failure penalty
            
        # Append histories
        self.state_history.append(np.copy(self.state))
        self.wind_history.append(float(np.sqrt(wind_roll**2 + wind_pitch**2)))
        
        info = {
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "failed": self.failed,
            "wind": float(np.sqrt(wind_roll**2 + wind_pitch**2))
        }
        
        return self._get_observation(), reward, done, info


class StreamingClassificationEnv:
    """
    Continual Image Classification under Distribution Shift (Task C).
    Synthesizes streaming features in R^64 representing CIFAR-10-C drift.
    15 corruption types are presented sequentially, each for 50 steps.
    """
    def __init__(self, classes: int = 10, d_in: int = 64, steps_per_corruption: int = 50):
        self.classes = classes
        self.d_in = d_in
        self.steps_per_corruption = steps_per_corruption
        
        # Define 10 class centers in R^64 representing standard normal features
        np.random.seed(42)
        self.class_centers = np.random.uniform(-1.0, 1.0, size=(classes, d_in))
        # Ensure class centers are unit norm
        self.class_centers = self.class_centers / np.linalg.norm(self.class_centers, axis=1, keepdims=True)
        
        # 15 Corruption Types
        self.corruptions = [
            "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
            "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
            "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"
        ]
        self.reset()

    def reset(self):
        self.current_step = 0
        self.total_steps = len(self.corruptions) * self.steps_per_corruption
        return self._get_next_sample()

    def _get_next_sample(self) -> Tuple[np.ndarray, int, str, int]:
        # Determine current corruption index and severity level
        corr_idx = self.current_step // self.steps_per_corruption
        if corr_idx >= len(self.corruptions):
            corr_idx = len(self.corruptions) - 1
            
        corr_name = self.corruptions[corr_idx]
        
        # Severity increases slowly during the corruption block from 1 to 5
        block_step = self.current_step % self.steps_per_corruption
        severity = 1 + int(block_step / (self.steps_per_corruption / 5))
        severity = min(5, severity)
        
        # Select random class label
        label = np.random.randint(0, self.classes)
        center = self.class_centers[label]
        
        # Synthesize distribution shift based on corruption type and severity
        # We apply systematic offsets and noise to the features
        shift_dir = np.sin(np.arange(self.d_in) * (corr_idx + 1))
        shift_dir = shift_dir / np.linalg.norm(shift_dir)
        
        # Shift magnitude increases with severity
        shift_magnitude = 0.1 * severity
        shifted_center = center + shift_dir * shift_magnitude
        
        # Add random intra-class variance (noise)
        noise = np.random.normal(0, 0.05, size=self.d_in)
        feature = shifted_center + noise
        
        # Ensure the feature is scaled properly
        feature = feature / np.linalg.norm(feature)
        
        return feature.astype(np.float32), label, corr_name, severity

    def step(self, action_logits: torch.Tensor) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        action_logits: logits from the swarm classification head, shape (10,)
        """
        # Get labels and stats for current step
        feature, label, corr_name, severity = self._get_next_sample()
        
        # Calculate accuracy
        pred = torch.argmax(action_logits).item()
        correct = (pred == label)
        
        self.current_step += 1
        done = self.current_step >= self.total_steps
        
        # Reward is 1.0 for correct classification, 0.0 otherwise
        reward = 1.0 if correct else 0.0
        
        info = {
            "correct": correct,
            "label": label,
            "pred": pred,
            "corruption": corr_name,
            "severity": severity,
            "step": self.current_step
        }
        
        # Get observation for next step
        if not done:
            next_feat, _, _, _ = self._get_next_sample()
        else:
            next_feat = np.zeros(self.d_in, dtype=np.float32)
            
        return next_feat, reward, done, info
