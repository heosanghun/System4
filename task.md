# Task Checklist: Latency Optimization Bridge Plan

- [x] **Phase 1: Refactor solver.py for Synchronization-Free Solving**
  - [x] Add `fixed_iter` support to `BroydenSolver.solve` in `system4/solver.py`
  - [x] Bypass `.item()` conversions and dynamic convergence checks under `fixed_iter` mode
- [x] **Phase 2: Refactor swarm.py to Eliminate CUDA-CPU Barriers**
  - [x] Update `update_mahalanobis_monitor` to perform operations strictly on GPU
  - [x] Pass conditional `fixed_iter` limits based on `z_prev` warm-start in `System4Swarm.forward`
- [x] **Phase 3: Integrate JIT Compilation (`torch.compile`)**
  - [x] Modify `system4/evaluate.py` to compile `swarm` and run warm-up iterations
  - [x] Modify `dashboard.py` to use compiled `swarm`
  - [x] Modify `visualize.py` to use compiled `swarm`
- [x] **Phase 4: Run Verification, Generate Plots, and Sync/Push**
  - [x] Run benchmark to verify System 4 median latency drops to sub-10ms
  - [x] Regenerate trajectory plots using `visualize.py`
  - [x] Copy plots and walkthrough back to brain and workspace directories
  - [x] Push all optimized modifications to GitHub
