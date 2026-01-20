"""
Monte Carlo demo: estimate the area under y = sin(πx) on [0,1]
using random sampling.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from mc_core import (
    curve,
    true_area,
    generate_points,
    running_estimate,
    choose_n_values,
    estimate_at_n,
    downsample_indices,
)


SIDEBAR_STYLE = """
<style>
section[data-testid="stSidebar"] {
  background-color: #f8f9fb;
}
section[data-testid="stSidebar"] > div {
  padding-top: 1.2rem;
}
section[data-testid="stSidebar"] h2 {
  font-size: 0.95rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #6c757d;
  margin-bottom: 0.4rem;
}
section[data-testid="stSidebar"] label {
  font-size: 0.9rem;
}
</style>
"""

st.sidebar.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)

def init_area_state() -> None:
    if "x_all" not in st.session_state:
        st.session_state.x_all = np.array([])
        st.session_state.y_all = np.array([])
        st.session_state.under_all = np.array([], dtype=bool)
        st.session_state.last_seed_on = None
        st.session_state.last_seed = None
        st.session_state.rng = None
    if "area_n_current" not in st.session_state:
        st.session_state.area_n_current = 0


# Initialize state for this page
init_area_state()

# Constants
TRUE_VALUE = true_area()

# Title and description
st.title("Monte Carlo Demo: Area under y = sin(πx) on [0,1]")
st.markdown(
    "We want to estimate the area under the curve without doing analytic calculus, "
    "purely by random sampling. The trick is to reinterpret this **area** as a "
    "**probability**."
)

st.markdown("**Define the target area:**")
st.latex(r"A = \int_0^1 \sin(\pi x)\,dx")

st.markdown(
    "Sample points (X, Y) uniformly in the unit square [0,1]×[0,1], and define the indicator:"
)
st.latex(r"I = \mathbf{1}\{Y < \sin(\pi X)\}")

st.markdown(
    "I equals 1 exactly when the random point lands under the curve. "
    "Uniform sampling means the probability of that event equals the area under the curve:"
)
st.latex(r"\mathbb{P}(Y < \sin(\pi X)) = A")
st.latex(r"A = \mathbb{E}[I]")

st.markdown("**Monte Carlo estimator (sample mean of indicators):**")
st.latex(
    r"\hat A_n = \frac{1}{n}\sum_{i=1}^n I_i"
    r" \;=\; \frac{1}{n}\sum_{i=1}^n \mathbf{1}\{y_i < \sin(\pi x_i)\}"
)

st.markdown("**True value for comparison:**")
st.latex(r"\int_0^1 \sin(\pi x)\,dx = \frac{2}{\pi} \approx 0.6366")

st.markdown("#### Why this estimator is correct (unbiased) and how fast it converges")
st.markdown(
    "- Each Ii is a Bernoulli random variable with P(Ii = 1) = A, so E[Ii] = A.\n"
    "- The sample mean therefore satisfies E[Ân] = A (unbiased).\n"
    "- Variance shrinks like 1/n, so typical error shrinks like 1/√n."
)
st.latex(r"\mathbb{E}[\hat A_n] = A")
st.latex(r"\mathrm{Var}(I_i) = A(1-A)")
st.latex(r"\mathrm{Var}(\hat A_n) = \frac{A(1-A)}{n}")
st.markdown("For large n, the Central Limit Theorem suggests an approximate normal error:")
st.latex(r"\hat A_n \approx \mathcal{N}\!\left(A,\; \frac{A(1-A)}{n}\right)")
st.markdown("So the typical scale of the error shrinks like 1/√n.")

# Sidebar controls (show below the Navigation section)
st.sidebar.header("Run Controls")

n_target = st.sidebar.number_input(
    "Target n (Finish runs until here)",
    min_value=0,
    max_value=500000,
    value=10000,
    step=100,
    key="area_n_target",
    help="Finish will generate points until the run reaches this many total samples.",
)

col_b1, col_b2, col_b3, col_b4 = st.sidebar.columns(4)
next_clicked = col_b1.button("Next", type="primary", key="area_next")
add100_clicked = col_b2.button("+100", key="area_add100")
finish_clicked = col_b3.button("Finish", key="area_finish")
reset_clicked = col_b4.button("Reset", key="area_reset")

st.sidebar.markdown("---")
st.sidebar.header("Randomness")

seed_on = st.sidebar.checkbox(
    "Use fixed seed (repeatable)", value=False, key="area_seed_on"
)

seed_value = None
if seed_on:
    seed_value = st.sidebar.number_input(
        "Seed value",
        min_value=0,
        max_value=2147483647,
        value=42,
        step=1,
        help="Random seed for reproducibility",
        key="area_seed_value",
    )

# Display controls
st.sidebar.header("Display Controls")

show_points = st.sidebar.checkbox(
    "Show random dots", value=True, key="area_show_points"
)

max_display = st.sidebar.slider(
    "max_display (render cap)",
    min_value=500,
    max_value=50000,
    value=5000,
    step=500,
    help="Maximum number of points to render in the scatter plot (downsampling for speed)",
    key="area_max_display",
)

# Convergence controls
st.sidebar.header("Convergence Controls")

show_convergence = st.sidebar.checkbox(
    "Show convergence plot", value=True, key="area_show_convergence"
)

n_max = st.sidebar.slider(
    "n_max (convergence max n)",
    min_value=1000,
    max_value=500000,
    value=10000,
    step=1000,
    help="Maximum n value used in the convergence curve",
    key="area_n_max",
)

steps = st.sidebar.slider(
    "steps (convergence points)",
    min_value=10,
    max_value=300,
    value=50,
    step=5,
    help="Number of points/markers on the convergence curve",
    key="area_steps",
)

# --- Stepwise state management ---
seed_changed = (
    seed_on != st.session_state.last_seed_on
    or (seed_on and seed_value != st.session_state.last_seed)
)

if reset_clicked or seed_changed:
    # Reset run (also resets RNG stream if seed changed)
    st.session_state.x_all = np.array([])
    st.session_state.y_all = np.array([])
    st.session_state.under_all = np.array([], dtype=bool)
    st.session_state.area_n_current = 0

# Ensure RNG is initialized
if seed_changed or (st.session_state.rng is None):
    if seed_on:
        st.session_state.rng = np.random.default_rng(seed_value)
        st.session_state.last_seed_on = seed_on
        st.session_state.last_seed = seed_value
    else:
        st.session_state.rng = np.random.default_rng()
        st.session_state.last_seed_on = seed_on
        st.session_state.last_seed = None

n_current = int(st.session_state.area_n_current)

n_to_add = 0
if next_clicked:
    n_to_add = 1
elif add100_clicked:
    n_to_add = 100
elif finish_clicked:
    n_to_add = max(0, int(n_target) - n_current)

if n_to_add > 0:
    x_new, y_new, under_new = generate_points(st.session_state.rng, n_to_add)
    st.session_state.x_all = np.concatenate([st.session_state.x_all, x_new])
    st.session_state.y_all = np.concatenate([st.session_state.y_all, y_new])
    st.session_state.under_all = np.concatenate(
        [st.session_state.under_all, under_new]
    )
    st.session_state.area_n_current = n_current + n_to_add

# Clamp current n to available data
n_current = min(int(st.session_state.area_n_current), len(st.session_state.under_all))

# Compute current estimate
n_use = n_current
estimate = estimate_at_n(st.session_state.under_all, n_use)
error = estimate - TRUE_VALUE

# Display metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Estimate (n={:,})".format(n_use), f"{estimate:.6f}")
with col2:
    st.metric("True Value (2/π)", f"{TRUE_VALUE:.6f}")
with col3:
    st.metric("Error", f"{error:.6f}", delta=None)

# Main plot area: two columns
col_left, col_right = st.columns(2)

# LEFT COLUMN: Unit-square picture
with col_left:
    st.subheader("Unit-square picture")

    if len(st.session_state.x_all) > 0:
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot the curve
        x_curve = np.linspace(0, 1, 1000)
        y_curve = curve(x_curve)
        ax.plot(x_curve, y_curve, "b-", linewidth=2, label="y = sin(πx)")

        # Shade area under the curve
        ax.fill_between(
            x_curve, 0, y_curve, alpha=0.2, color="blue", label="Area under curve"
        )

        # Scatter points (downsample if needed)
        n_show = min(n_use, len(st.session_state.x_all))
        if show_points and n_show > 0:
            # Determine which points to display
            if n_show <= max_display:
                indices = np.arange(n_show)
            else:
                indices = downsample_indices(n_show, max_display)

            x_display = st.session_state.x_all[indices]
            y_display = st.session_state.y_all[indices]
            under_display = st.session_state.under_all[indices]

            # Separate points by whether they're under the curve
            under_mask = under_display
            above_mask = ~under_display

            if np.any(under_mask):
                ax.scatter(
                    x_display[under_mask],
                    y_display[under_mask],
                    c="green",
                    s=10,
                    alpha=0.5,
                    label="Under curve",
                    edgecolors="none",
                )
            if np.any(above_mask):
                ax.scatter(
                    x_display[above_mask],
                    y_display[above_mask],
                    c="red",
                    s=10,
                    alpha=0.5,
                    label="Above curve",
                    edgecolors="none",
                )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title(
            f"Monte Carlo Estimate: {estimate:.6f} (n={n_use:,})", fontsize=11
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Click 'Generate / Resample points' to start the simulation.")

# RIGHT COLUMN: Convergence plot
with col_right:
    st.subheader("Convergence as n increases")

    if show_convergence and (len(st.session_state.under_all) > 0):
        if len(st.session_state.under_all) > 0:
            # Compute convergence curve
            n_vals = choose_n_values(min(n_max, n_use), steps)
            estimates_conv = np.array(
                [estimate_at_n(st.session_state.under_all, n) for n in n_vals]
            )

            fig, ax = plt.subplots(figsize=(6, 6))

            # Plot convergence
            ax.plot(
                n_vals,
                estimates_conv,
                "b-",
                linewidth=1.5,
                alpha=0.7,
                label="Running estimate",
            )
            ax.scatter(
                n_vals, estimates_conv, s=30, c="blue", alpha=0.6, zorder=5
            )

            # Horizontal line at true value
            ax.axhline(
                TRUE_VALUE,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"True value = {TRUE_VALUE:.6f}",
            )

            # Vertical line and marker at current n
            if n_use > 0 and n_use <= n_max:
                est_current = estimate_at_n(st.session_state.under_all, n_use)
                ax.axvline(
                    n_use,
                    color="orange",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label=f"current n = {n_use:,}",
                )
                ax.scatter(
                    [n_use],
                    [est_current],
                    s=100,
                    c="orange",
                    marker="*",
                    zorder=10,
                    edgecolors="black",
                    linewidths=1,
                )

            ax.set_xscale("log")
            ax.set_xlabel("n (number of points)", fontsize=12)
            ax.set_ylabel("Estimate", fontsize=12)
            ax.set_title("Convergence of Monte Carlo Estimate", fontsize=11)
            ax.grid(True, alpha=0.3, which="both")
            ax.legend(loc="best", fontsize=9)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Generate points to see the convergence plot.")
    elif not show_convergence:
        st.info(
            "Enable 'Show convergence plot' in the sidebar to display this plot."
        )
    else:
        st.info("Click 'Generate / Resample points' to start the simulation.")

# Footer
st.markdown("---")
st.markdown(
    "*Use the sidebar controls to adjust simulation parameters and visualizations.*"
)

