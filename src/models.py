from numba import njit
import numpy as np
import random

@njit
def _weighted_choice_numba(weights):
    """Pick an index proportionally to weights (Numba-compatible)."""
    total = np.sum(weights)
    r = np.random.random() * total
    upto = 0.0
    for i in range(weights.shape[0]):
        upto += weights[i]
        if upto >= r:
            return i
    return weights.shape[0] - 1

def unconstrained_simulation(L=50, p=0.3, t_max=100.0, seed=None):
    """Kinetic Monte Carlo for an unconstrained spin-flip process."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    config = np.random.binomial(1, p, size=L)

    t = 0.0
    times = [t]
    densities = [np.mean(config)]

    rates = np.array([p if config[i] == 0 else 1 - p for i in range(L)], dtype=float)

    while t < t_max:
        total_rate = np.sum(rates)
        if total_rate == 0:
            break

        delta_t = np.random.exponential(scale=1 / total_rate)
        t += delta_t
        times.append(t)

        probabilities = rates / total_rate
        x = np.random.choice(L, p=probabilities)

        config[x] = 1 - config[x]
        rates[x] = p if config[x] == 0 else 1 - p
        densities.append(np.mean(config))

    return np.array(times), np.array(densities)

@njit
def simulation_east(L=1000, p=0.7, t_max=100.0, seed=0, initial=0.0, N_points=500):
    if seed != 0:
        np.random.seed(seed)

    fixed_times = np.linspace(0.0, t_max, N_points)

    config = np.random.binomial(1, initial, size=L)
    t = 0.0
    densities = np.zeros(N_points)
    record_index = 0

    active = np.zeros(L, dtype=np.uint8)
    rates = np.zeros(L, dtype=np.float64)

    for i in range(L):
        right = config[i + 1] if i < L - 1 else 1
        if right == 0:
            active[i] = 1
            rates[i] = p if config[i] == 0 else 1 - p
        else:
            active[i] = 0
            rates[i] = 0.0

    while t < t_max and np.any(active) and record_index < N_points:
        total_rate = np.sum(rates)
        if total_rate == 0.0:
            break

        dt = np.random.exponential(1.0 / total_rate)
        t += dt

        x = _weighted_choice_numba(rates)
        config[x] = 1 - config[x]

        for j in (x - 1, x, x + 1):
            if 0 <= j < L:
                right = config[j + 1] if j < L - 1 else 1
                if right == 0:
                    active[j] = 1
                    rates[j] = p if config[j] == 0 else 1 - p
                else:
                    active[j] = 0
                    rates[j] = 0.0

        while record_index < N_points and t >= fixed_times[record_index]:
            densities[record_index] = np.mean(config)
            record_index += 1

    return densities

@njit
def simulation_fa1f(L=1000, p=0.7, t_max=100.0, seed=0, initial=0.0, N_points=50):
    if seed != 0:
        np.random.seed(seed)

    fixed_times = np.linspace(0.0, t_max, N_points)
    config = np.random.binomial(1, initial, size=L)
    t = 0.0
    densities = np.zeros(N_points)
    record_index = 0

    active = np.zeros(L, dtype=np.uint8)
    rates = np.zeros(L, dtype=np.float64)

    for i in range(L):
        left = config[i - 1] if i > 0 else 1
        right = config[i + 1] if i < L - 1 else 1
        if left == 0 or right == 0:
            active[i] = 1
            rates[i] = p if config[i] == 0 else 1 - p
        else:
            active[i] = 0
            rates[i] = 0.0

    while t < t_max and np.any(active) and record_index < N_points:
        total_rate = np.sum(rates)
        if total_rate == 0.0:
            break

        dt = np.random.exponential(1.0 / total_rate)
        t += dt
        x = _weighted_choice_numba(rates)
        config[x] = 1 - config[x]

        for j in (x - 1, x, x + 1):
            if 0 <= j < L:
                left = config[j - 1] if j > 0 else 1
                right = config[j + 1] if j < L - 1 else 1
                if left == 0 or right == 0:
                    active[j] = 1
                    rates[j] = p if config[j] == 0 else 1 - p
                else:
                    active[j] = 0
                    rates[j] = 0.0

        while record_index < N_points and t >= fixed_times[record_index]:
            densities[record_index] = np.mean(config)
            record_index += 1

    return densities

@njit
def simulation_addictive(L=1000, p=0.7, t_max=100.0, seed=0, initial=0.0, N_points=50):
    if seed != 0:
        np.random.seed(seed)

    fixed_times = np.linspace(0.0, t_max, N_points)
    config = np.random.binomial(1, initial, size=L)
    t = 0.0
    densities = np.zeros(N_points)
    record_index = 0

    active = np.zeros(L, dtype=np.uint8)
    rates = np.zeros(L, dtype=np.float64)

    for i in range(L):
        left = config[i - 1] if i > 0 else 1
        right = config[i + 1] if i < L - 1 else 1
        empty_neighbors = (left == 0) + (right == 0)
        if empty_neighbors == 0:
            rate = 0.0
        elif empty_neighbors == 1:
            rate = (1 - p) / 2.0 if config[i] == 1 else p / 2.0
        else:
            rate = (1 - p) if config[i] == 1 else p
        if rate > 0.0:
            active[i] = 1
            rates[i] = rate
        else:
            active[i] = 0
            rates[i] = 0.0

    while t < t_max and np.any(active) and record_index < N_points:
        total_rate = np.sum(rates)
        if total_rate == 0.0:
            break

        t += np.random.exponential(1.0 / total_rate)
        x = _weighted_choice_numba(rates)
        config[x] = 1 - config[x]

        for j in range(max(0, x - 1), min(L, x + 2)):
            left = config[j - 1] if j > 0 else 1
            right = config[j + 1] if j < L - 1 else 1
            empty_neighbors = (left == 0) + (right == 0)
            if empty_neighbors == 0:
                rate = 0.0
            elif empty_neighbors == 1:
                rate = (1 - p) / 2.0 if config[j] == 1 else p / 2.0
            else:
                rate = (1 - p) if config[j] == 1 else p
            if rate > 0.0:
                active[j] = 1
                rates[j] = rate
            else:
                active[j] = 0
                rates[j] = 0.0

        while record_index < N_points and t >= fixed_times[record_index]:
            densities[record_index] = np.mean(config)
            record_index += 1

    return densities

@njit
def simulation_fa1f_full(L=1000, p=0.7, t_max=100.0, seed=0, initial=0.0, N_points=50):
    if seed != 0:
        np.random.seed(seed)

    fixed_times = np.linspace(0.0, t_max, N_points)

    config = np.ones(L, dtype=np.uint8)
    config[-1] = 0

    t = 0.0
    densities = np.zeros(N_points)
    record_index = 0

    active = np.zeros(L, dtype=np.uint8)
    rates = np.zeros(L, dtype=np.float64)

    for i in range(L):
        left = config[i - 1] if i > 0 else 1
        right = config[i + 1] if i < L - 1 else 1
        if left == 0 or right == 0:
            active[i] = 1
            rates[i] = p if config[i] == 0 else 1 - p
        else:
            active[i] = 0
            rates[i] = 0.0

    while t < t_max and np.any(active) and record_index < N_points:
        total_rate = np.sum(rates)
        if total_rate == 0.0:
            break

        dt = np.random.exponential(1.0 / total_rate)
        t += dt
        x = _weighted_choice_numba(rates)
        config[x] = 1 - config[x]

        for j in (x - 1, x, x + 1):
            if 0 <= j < L:
                left = config[j - 1] if j > 0 else 1
                right = config[j + 1] if j < L - 1 else 1
                if left == 0 or right == 0:
                    active[j] = 1
                    rates[j] = p if config[j] == 0 else 1 - p
                else:
                    active[j] = 0
                    rates[j] = 0.0

        while record_index < N_points and t >= fixed_times[record_index]:
            densities[record_index] = np.mean(config)
            record_index += 1

    return densities
