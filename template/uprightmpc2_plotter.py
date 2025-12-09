from uprightmpc2 import *
import os
import numpy as np
import matplotlib.pyplot as plt

# Base name; we’ll prepend model/solver
DATA_FILE_TEMPLATE = "{}_sweep_results.npz"

PLOT_DIR = "/Users/franklinho/Documents/Work, Research, Finance/Academic/2025-2026 Classes/APMTH 205/Final Project Plots"


def plot_from_file(model: str = "osqp"):
    """
    Plot sweep results for a given solver/model.

    Parameters
    ----------
    model : str
        Solver/model name; e.g. "osqp" or "qpoases".
    """
    model = model.lower()

    # Map to pretty label for plots
    if model == "osqp":
        solver_label = "OSQP"
    elif model in ("qpoases", "qp_oases", "qp-oases"):
        solver_label = "qpOASES"
        model = "qpoases"  # normalize for filenames
    else:
        solver_label = model.upper()

    data_file = DATA_FILE_TEMPLATE.format(model)

    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"{data_file} not found. Run the sweep for model '{model}' first to generate data."
        )

    os.makedirs(PLOT_DIR, exist_ok=True)

    data = np.load(data_file)
    vals         = data["vals"]
    avg_times_ms = data["avg_times_ms"]
    rms_errors   = data["rms_errors"]

    label_fs = 20
    tick_fs = 20

    tol_xlabel = rf"{solver_label} tolerance  $\epsilon_{{\mathrm{{abs}}}}$"
    time_ylabel = f"Average {solver_label} solve time [ms]"
    err_ylabel = "RMS tracking error [mm]"

    # --- 1) Avg solve time vs tolerance (x log, y linear) ---
    fig1, ax1 = plt.subplots()
    ax1.semilogx(vals, avg_times_ms, marker="o")
    ax1.set_xlabel(tol_xlabel, fontsize=label_fs)
    ax1.set_ylabel(time_ylabel, fontsize=label_fs)
    ax1.tick_params(axis="both", labelsize=tick_fs)
    ax1.grid(True, which="both")
    fig1.tight_layout()
    fig1.savefig(
        os.path.join(PLOT_DIR, f"avg_{model}_solve_time_vs_tol.png"),
        dpi=400,
        bbox_inches="tight",
    )
    plt.show()

    # --- 2) RMS tracking error vs tolerance (full, x log, y log) ---
    fig2, ax2 = plt.subplots()
    ax2.semilogx(vals, rms_errors, marker="o")
    ax2.set_yscale("log")  # NEW: log scale on y
    ax2.set_xlabel(tol_xlabel, fontsize=label_fs)
    ax2.set_ylabel(err_ylabel, fontsize=label_fs)
    ax2.tick_params(axis="both", labelsize=tick_fs)
    ax2.grid(True, which="both")
    fig2.tight_layout()
    fig2.savefig(
        os.path.join(PLOT_DIR, f"{model}_rms_trck_err_full.png"),
        dpi=400,
        bbox_inches="tight",
    )
    plt.show()

    # --- 3) RMS tracking error vs tolerance, zoomed (x log, y log) ---
    fig3, ax3 = plt.subplots()
    ax3.semilogx(vals, rms_errors, marker="o")
    ax3.set_yscale("log")  # NEW: log scale on y
    ax3.set_xlabel(tol_xlabel, fontsize=label_fs)
    ax3.set_ylabel(err_ylabel, fontsize=label_fs)
    ax3.tick_params(axis="both", labelsize=tick_fs)
    ax3.grid(True, which="both")

    x_min_limit = 1e-5
    x_max_limit = 1e-2
    ax3.set_xlim(x_min_limit, x_max_limit)

    # Adjust ylim based on data within the x window (must stay > 0 for log)
    mask = (vals >= x_min_limit) & (vals <= x_max_limit)
    y_visible = rms_errors[mask]

    ymin = y_visible.min()
    ymax = y_visible.max()

    # Use multiplicative padding (safer for log scale)
    if ymax > ymin > 0:
        ax3.set_ylim(0.8 * ymin, 1.2 * ymax)
    else:
        # Fallback if something weird happens
        ax3.set_ylim(1e-6, 10)

    fig3.tight_layout()
    fig3.savefig(
        os.path.join(PLOT_DIR, f"{model}_rms_trck_err_zoom.png"),
        dpi=400,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    plot_from_file("qpOASES")
