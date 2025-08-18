from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random
from tqdm import tqdm

def single_run(args):
    simulation_fn, L, p, t_max, initial, seed, N_points = args
    return simulation_fn(L=L, p=p, t_max=t_max, initial=initial, N_points=N_points, seed=seed)

def ave_simulation(
    simulation_fn=None,
    L=1000,
    p=0.7,
    t_max=100.0,
    seed=None,
    num_runs=100,
    N_points=500,
    initial=0.0,
):
    """Average `num_runs` simulations and return (times, mean_densities)."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    fixed_times = np.linspace(0.0, t_max, N_points)
    seeds = [0 if seed is None else seed + n for n in range(num_runs)]
    args_list = [(simulation_fn, L, p, t_max, initial, s, N_points) for s in seeds]

    results = []
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(single_run, args_list), total=num_runs, desc="Simulating"):
            results.append(result)

    all_densities = np.array(results)
    mean_densities = np.mean(all_densities, axis=0)
    return fixed_times, mean_densities
