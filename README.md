# Kinetically Constrained Models — M1 Internship

This repository contains simulations and analysis from my M1 internship. It includes implementations of several 1D kinetically constrained models (East, FA-1f, additive facilitation) and curve-fitting analyses to study relaxation and front dynamics.

## Structure
- `notebooks/` — exploration notebooks (original internship notebook lives here).
- `src/` — reusable Python modules split by the four main parts of the work.
- `data/`, `results/` — cached arrays and figures (ignored by git by default).

## Install
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start
```python
from src.models import simulation_fa1f_full
from src.simulation import ave_simulation

# Example: average 3 runs
times, dens = ave_simulation(
    simulation_fn=simulation_fa1f_full,
    L=500,
    p=0.1,
    t_max=3000.0,
    num_runs=3,
    N_points=500,
    initial=0.0,
)
```

## Reproducing figures
Use the plotting helpers in `src/fitting_basic.py` and `src/fitting_full_density.py`.

## License
MIT (see `LICENSE`).
