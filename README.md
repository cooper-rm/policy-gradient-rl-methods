# Lab 6: PyTorch and Actor-Critic Methods

**Morgan Cooper -- MSDS 684, Reinforcement Learning**

## Overview

This lab solves Gymnasium's Pendulum-v1 using an online actor-critic algorithm with TD(0) bootstrapping, implemented in PyTorch. The actor outputs a Gaussian policy over continuous torque actions and the critic learns a state-value function $V(s)$. Both networks are updated every step using the TD error as the advantage signal.

## Structure

- `Lab6_Actor_Critic.ipynb` -- Main notebook with implementation, training, and visualizations
- `generate_report.py` -- Generates the PDF report from LaTeX
- `Cooper_Morgan_Lab6.pdf` -- Final report
- `figures/` -- Saved visualizations used in the report (latest run + per-version archives)
- `requirements.txt` -- Python dependencies

## Key Components

- **Gaussian Actor**: Two hidden layers of 64 tanh units, outputs mean $\mu(s)$ tanh-squashed to the action range. Standard deviation is a learned scalar parameter clamped between $\sigma=0.13$ and $\sigma=0.50$.
- **Value Critic**: Same hidden architecture, outputs scalar $V(s)$.
- **Online TD(0) update**: Per step, $\delta = r + \gamma V(s') - V(s)$ is used as the critic loss ($\delta^2$) and detached as the advantage in the actor loss ($-\log\pi(a|s)\cdot\delta$).
- **Stability fixes**: Gradient clipping at 1.0, best-checkpoint restore, and balance-region reward shaping.

## Results

| Metric | Value |
|---|---|
| Mean final return (30 seeds) | -1210 |
| Standard deviation (30 seeds) | 363 |
| Best seed final return | -264 |
| Seeds that reached return > -500 | 4 of 30 |

The successful seeds learned a static balance at $\theta \approx -0.25$ rad with a constant torque around $+0.6$, where gravity is exactly cancelled by the applied torque.

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook end-to-end
jupyter notebook Lab6_Actor_Critic.ipynb

# Generate the PDF report
python generate_report.py
```

The notebook saves figures to `figures/` and version snapshots to `figures/v{RUN_VERSION}/`. Bump `RUN_VERSION` in the setup cell before each rerun to keep snapshots from overwriting each other.

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Chapter 13.
- Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS*.
