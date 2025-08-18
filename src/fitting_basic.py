import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fit_and_plot_regular_exp(times, densities):
    def regular_exp(t, A, B, lam):
        return A - B * np.exp(-lam * t)
    p0 = [float(np.max(densities)), 0.5, 0.5]
    popt, _ = curve_fit(regular_exp, times, densities, p0=p0, maxfev=5000)
    A_fit, B_fit, lam_fit = popt
    fit_curve = regular_exp(np.array(times), *popt)
    plt.plot(times, densities, label='Simulated')
    plt.plot(times, fit_curve, linestyle='--', label=f'f(t) = {A_fit:.3f} - {B_fit:.3f} e^(-{lam_fit:.3f} t)')
    plt.xlabel('Time'); plt.ylabel('Density of active sites'); plt.title('Exponential fit to density'); plt.legend(); plt.show()
    return popt

def fit_and_plot_stretched_exp(times, densities):
    def stretched_exp(t, A, B, lam, beta):
        return A - B * np.exp(-lam * t**beta)
    p0 = [float(np.max(densities)), 0.5, 0.5, 0.5]
    popt, _ = curve_fit(stretched_exp, times, densities, p0=p0, maxfev=5000)
    A_fit, B_fit, lam_fit, beta_fit = popt
    fit_curve = stretched_exp(np.array(times), *popt)
    plt.plot(times, densities, label='Simulated')
    plt.plot(times, fit_curve, linestyle='--', label=f'f(t) = {A_fit:.3f} - {B_fit:.3f} e^(-{lam_fit:.3f} t^{beta_fit:.3f})')
    plt.xlabel('Time'); plt.ylabel('Density of active sites'); plt.title('Stretched exponential fit'); plt.legend(); plt.show()
    return popt

def plot_and_fit_log_density_alpha0(times, densities, q):
    transformed = np.abs(1 - q - np.array(densities))
    mask = transformed > 0
    t_trans = np.array(times)[mask]
    y_log = np.log(transformed[mask])
    mask_trim = t_trans < (times[-1] if len(times) else 0)
    t_trans = t_trans[mask_trim]
    y_log = y_log[mask_trim]
    def linear_func(t, a, b): return a * t + b
    popt, _ = curve_fit(linear_func, t_trans, y_log)
    a_fit, b_fit = popt
    plt.plot(t_trans, y_log, 'o', label='log(|1 - q - density|)')
    plt.plot(t_trans, linear_func(t_trans, *popt), linestyle='--', label=f'Fit: y={a_fit:.4f} t + {b_fit:.4f}')
    plt.xlabel('t'); plt.ylabel('log(|1 - q - density|)'); plt.title('Alpha=0 linearization'); plt.legend(); plt.grid(True); plt.show()
    return popt

def plot_and_fit_log_density_alpha1(times, densities, q):
    transformed = np.abs((1 - q - np.array(densities)) / (1 - np.array(densities)))
    mask = transformed > 0
    t_trans = np.array(times)[mask]
    y_log = np.log(transformed[mask])
    mask_trim = t_trans > 10
    t_trans = t_trans[mask_trim]
    y_log = y_log[mask_trim]
    def linear_func(t, a, b): return a * t + b
    popt, _ = curve_fit(linear_func, t_trans, y_log)
    a_fit, b_fit = popt
    plt.plot(t_trans, y_log, 'o', label='log(|(1-q-ρ)/(1-ρ)|)')
    plt.plot(t_trans, linear_func(t_trans, *popt), linestyle='--', label=f'Fit: y={a_fit:.4f} t + {b_fit:.4f}')
    plt.xlabel('t'); plt.ylabel('log(|(1-q-ρ)/(1-ρ)|)'); plt.title('Alpha=1 linearization'); plt.legend(); plt.grid(True); plt.show()
    return popt

def plot_and_fit_modified_alpha2(times, densities, p):
    times = np.array(times)
    densities = np.array(densities)
    numerator = np.abs(p - densities)
    denominator = 1 - densities
    transformed = np.log(numerator / denominator) + (1 - p) / denominator
    mask = (numerator > 0) & (denominator > 0)
    t_valid = times[mask]
    y_valid = transformed[mask]
    mask_trim = t_valid > 10
    t_no_out = t_valid[mask_trim]
    y_no_out = y_valid[mask_trim]
    def linear_func(t, C1, C2): return C1 * t + C2
    popt, _ = curve_fit(linear_func, t_no_out, y_no_out)
    C1_fit, C2_fit = popt
    plt.plot(t_no_out, y_no_out, 'o', label='Modified α=2 transform')
    plt.plot(t_no_out, linear_func(t_no_out, *popt), linestyle='--', label=f'Fit: y={C1_fit:.3f} t + {C2_fit:.3f}')
    plt.xlabel('t'); plt.ylabel('transform'); plt.title('Alpha=2 modified linearization'); plt.legend(); plt.grid(True); plt.show()
    return popt

def plot_custom_expression_alpha3(times, densities, q):
    p = 1 - q
    rho_t = np.array(densities)
    t_vals = np.array(times)
    safe_mask = (1 - rho_t) != 0
    safe_mask &= np.abs((p - rho_t) / (1 - rho_t)) > 0
    rho_t = rho_t[safe_mask]
    t_vals = t_vals[safe_mask]
    term1 = (2.0 / p) * np.log(np.abs((p - rho_t) / (1 - rho_t)))
    term2 = 1.0 / (2.0 * (p - 1.0) * (1.0 - rho_t) ** 2)
    term3 = 1.0 / (1.0 - rho_t)
    y_vals = term1 - term2 + term3
    def linear_func(t, a, b): return a * t + b
    popt, _ = curve_fit(linear_func, t_vals, y_vals)
    a_fit, b_fit = popt
    plt.plot(t_vals, y_vals, 'o', label='custom α=3 expression')
    plt.plot(t_vals, linear_func(t_vals, *popt), linestyle='--', label=f'Fit: y={a_fit:.3f} t + {b_fit:.3f}')
    plt.xlabel('t'); plt.ylabel('expression'); plt.title('Alpha=3 custom linearization'); plt.legend(); plt.grid(True); plt.show()
    return popt

def plot_fa1f_log_expression(times, densities, p=0.5):
    times = np.array(times)
    densities = np.array(densities)
    num = (1 + densities) ** (1 - p) * (1 - densities) ** (1 + p)
    denom = (p - densities) ** 2
    expr = num / denom
    mask = (expr > 0) & np.isfinite(expr)
    t_final = times[mask]
    y_final = np.log(expr[mask])
    if len(t_final) < 2:
        raise ValueError("Not enough valid data to fit")
    def linear_func(t, a, b): return a * t + b
    popt, _ = curve_fit(linear_func, t_final, y_final)
    a_fit, b_fit = popt
    plt.plot(t_final, y_final, 'o', label='FA-1f log expr')
    plt.plot(t_final, linear_func(t_final, *popt), linestyle='--', label=f'Fit: y={a_fit:.3f} t + {b_fit:.3f}')
    plt.xlabel('t'); plt.ylabel('log(expr)'); plt.title('FA-1f log-transformed expression'); plt.legend(); plt.grid(True); plt.show()
    return popt
