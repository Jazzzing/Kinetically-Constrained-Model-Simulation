import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from .simulation import ave_simulation

def piecewise_smooth(t, y0, m, t1, y_final, tau):
    t = np.array(t)
    linear = y0 + m * t
    smooth = y_final + (linear - y_final) * np.exp(-(t - t1) / tau)
    return np.where(t < t1, linear, smooth)

def fit_and_plot_smooth(times, densities):
    y0_guess = float(densities[0])
    y_final_guess = float(densities[-1])
    t1_guess = float(times[len(times) // 2])
    m_guess = float((densities[-1] - densities[0]) / (times[-1] - times[0]))
    tau_guess = float((times[-1] - times[0]) / 10.0)
    p0 = [y0_guess, m_guess, t1_guess, y_final_guess, tau_guess]
    popt, _ = curve_fit(piecewise_smooth, times, densities, p0=p0, maxfev=20000)
    y0_fit, m_fit, t1_fit, y_final_fit, tau_fit = popt
    fit_curve = piecewise_smooth(times, *popt)
    threshold = abs(y0_fit - y_final_fit) * 0.001
    idx_const = np.argmax(np.abs(fit_curve - y_final_fit) <= threshold)
    t_const = times[idx_const] if idx_const > 0 else None
    plt.plot(times, densities, label='Simulated data')
    plt.plot(times, fit_curve, linestyle='--', label='Smooth 2-Segment Fit')
    plt.axvline(t1_fit, linestyle=':', label=f't1={t1_fit:.0f}')
    if t_const is not None:
        plt.axvline(t_const, linestyle='--', label=f'Const start ≈ {t_const:.1f}')
    plt.xlabel('Time'); plt.ylabel('Density of active sites'); plt.title('Linear + Smooth Transition to Plateau'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
    return popt, t_const

def estimate_gradient(times, densities):
    y0_guess = float(densities[0])
    y_final_guess = float(densities[-1])
    t1_guess = float(times[len(times)//2])
    m_guess = float((densities[-1] - densities[0]) / (times[-1] - times[0]))
    tau_guess = float((times[-1] - times[0]) / 10.0)
    p0 = [y0_guess, m_guess, t1_guess, y_final_guess, tau_guess]
    popt, pcov = curve_fit(piecewise_smooth, times, densities, p0=p0, maxfev=20000)
    y0_fit, m_fit, t1_fit, y_final_fit, tau_fit = popt
    return {
        "y0": y0_fit, "m": m_fit, "t1": t1_fit, "y_final": y_final_fit, "tau": tau_fit,
        "cov": pcov,
    }

def sweep_L_for_m(
    simulation_fn,
    L_values,
    *,
    p=0.7,
    t_max=100.0,
    N_points=500,
    num_runs=100,
    base_seed=42,
    initial=0.0,
):
    rows = []
    for L in L_values:
        seed = None if base_seed is None else base_seed + (hash(("L", L, "p", p)) % 10**6)
        times, dens = ave_simulation(
            simulation_fn=simulation_fn,
            L=L, p=p, t_max=t_max, seed=seed,
            num_runs=num_runs, N_points=N_points, initial=initial,
        )
        fit = estimate_gradient(times, dens)
        v = -L * fit['m'] / (1.0 - p)
        rows.append({
            "L": L,
            "p": p,
            "m": fit["m"],
            "v": v,
            "y0": fit["y0"],
            "t1": fit["t1"],
            "y_final": fit["y_final"],
            "tau": fit["tau"],
        })
    import pandas as pd
    return pd.DataFrame(rows).sort_values("L").reset_index(drop=True)

def plot_v_vs_L(df):
    import matplotlib.pyplot as plt
    sub = df.sort_values("L")
    plt.figure()
    plt.plot(sub["L"], sub["v"], marker="o")
    plt.xlabel("L"); plt.ylabel("velocity v (from fit)"); plt.title("Velocity v vs L"); plt.grid(True); plt.tight_layout(); plt.show()

def sweep_p_for_m(
    simulation_fn,
    p_values,
    *,
    L=1000,
    t_max=100.0,
    N_points=500,
    num_runs=100,
    base_seed=42,
    initial=0.0,
):
    rows = []
    for p in p_values:
        if np.isclose(p, 1.0):
            raise ValueError("p=1 leads to division by zero in v = -L*m/(1-p).")
        seed = None if base_seed is None else base_seed + (hash(("L", L, "p", p)) % 10**6)
        times, dens = ave_simulation(
            simulation_fn=simulation_fn,
            L=L, p=p, t_max=t_max, seed=seed,
            num_runs=num_runs, N_points=N_points, initial=initial,
        )
        fit = estimate_gradient(times, dens)
        v = -L * fit["m"] / (1.0 - p)
        rows.append({
            "L": L,
            "p": p,
            "m": fit["m"],
            "v": v,
            "y0": fit["y0"],
            "t1": fit["t1"],
            "y_final": fit["y_final"],
            "tau": fit["tau"],
        })
    import pandas as pd
    return pd.DataFrame(rows).sort_values("p").reset_index(drop=True)

def plot_v_vs_p(df):
    import matplotlib.pyplot as plt
    sub = df.sort_values("p")
    plt.figure()
    plt.plot(sub["p"], sub["v"], marker="o")
    plt.xlabel("p"); plt.ylabel("velocity v = -L*m/(1-p)"); plt.title("Velocity v vs p"); plt.grid(True); plt.tight_layout(); plt.show()
