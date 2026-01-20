"""
Buffon's Needle simulation: estimate π from random needle drops.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from mc_core import choose_n_values


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

def init_buffon_state() -> None:
    if "buffon_theta" not in st.session_state:
        st.session_state.buffon_theta = np.array([])
        st.session_state.buffon_x = np.array([])
        st.session_state.buffon_crosses = np.array([], dtype=bool)
        st.session_state.buffon_n_current = 0
        st.session_state.buffon_last_seed_on = None
        st.session_state.buffon_last_seed = None
        st.session_state.buffon_rng = None


# Initialize state for this page
init_buffon_state()

st.title("Buffon's Needle: Estimating π")
st.markdown(
    "Buffon's Needle is a geometric probability experiment that lets us estimate π "
    "from random line crossings. Imagine an infinite floor with parallel lines, each "
    "distance d apart. We repeatedly drop a needle of length L at random positions "
    "and orientations, and record whether it crosses a line."
)

st.markdown(
    "For the classical version, we assume **L ≤ d**. Each drop is described by two "
    "random quantities:"
)
st.markdown(
    "- **Θ**: the acute angle between the needle and the parallel lines (uniform by symmetry)"
)
st.markdown(
    "- **X**: the distance from the needle’s center to the nearest line (uniform by position)"
)
st.markdown("These are assumed independent under the “random toss” model.")

st.markdown("**Distributions (written explicitly):**")
st.latex(r"\Theta \sim \mathrm{Uniform}\!\left(0,\frac{\pi}{2}\right)")
st.latex(r"X \sim \mathrm{Uniform}\!\left(0,\frac{d}{2}\right)")

st.markdown("**Crossing condition (for $L \\le d$):**")
st.latex(r"X \le \frac{L}{2}\,\sin(\Theta)")

st.markdown(
    "Geometrically, (L/2)·sin(Θ) is the perpendicular distance from the needle’s "
    "center to one of its endpoints. The needle crosses a line exactly when that "
    "perpendicular reach is at least X."
)

st.markdown("#### Deriving the crossing probability")
st.markdown(
    "Condition on Θ = θ. Given that angle, the needle crosses whenever "
    "X ≤ (L/2)·sin(θ). Since X is uniform on [0, d/2]:"
)
st.latex(
    r"\mathbb{P}(\text{cross} \mid \Theta=\theta)"
    r" = \frac{\min\left(\frac{L}{2}\sin\theta,\; \frac{d}{2}\right)}{\frac{d}{2}}."
)

st.markdown("When L ≤ d, we always have (L/2)·sin(θ) ≤ d/2, so this simplifies to:")
st.latex(
    r"\mathbb{P}(\text{cross} \mid \Theta=\theta)"
    r" = \frac{L}{d}\,\sin\theta."
)

st.markdown("Now average over the random angle Θ, which is uniform on [0, π/2]:")
st.latex(
    r"\mathbb{P}(\text{cross})"
    r" = \mathbb{E}\big[\mathbb{P}(\text{cross}\mid\Theta)\big]"
    r" = \frac{2}{\pi}\int_0^{\pi/2} \frac{L}{d}\sin\theta\,d\theta"
    r" = \frac{2L}{\pi d}."
)

st.markdown("So for $L \\le d$ we obtain the famous identity:")
st.latex(r"P(\text{cross}) = \frac{2L}{\pi d}")

st.markdown("**Monte Carlo estimator for $\\pi$:**")
st.latex(
    r"\hat{P} = \frac{\#\text{crosses}}{n},"
    r"\qquad"
    r"\hat{\pi} = \frac{2L}{d\,\hat{P}}."
)

st.markdown(
    "Here P̂ is just the sample fraction of drops that cross a line. "
    "If the identity P(cross) = 2L/(πd) holds, rearranging for π gives the estimator above. "
    "As n grows, P̂ concentrates around the true probability (Law of Large Numbers), "
    "and π̂ converges toward π."
)
st.markdown(
    "Because π̂ involves 1/P̂, runs with very few crossings (small P̂) can be noisy. "
    "Choosing L closer to d tends to produce more crossings and lower variance in practice."
)

# Sidebar controls
st.sidebar.header("Run Controls")

n_target = st.sidebar.number_input(
    "Target n (Finish runs until here)",
    min_value=0,
    max_value=200000,
    value=10000,
    step=100,
    key="buffon_n_target",
    help="Finish will simulate drops until the run reaches this many total needles.",
)

col_b1, col_b2, col_b3, col_b4 = st.sidebar.columns(4)
next_clicked = col_b1.button("Next", type="primary", key="buffon_next")
add100_clicked = col_b2.button("+100", key="buffon_add100")
finish_clicked = col_b3.button("Finish", key="buffon_finish")
reset_clicked = col_b4.button("Reset", key="buffon_reset")

st.sidebar.markdown("---")
st.sidebar.header("Parameters")

# Geometry parameters
d = st.sidebar.slider(
    "Line spacing d",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.1,
    key="buffon_d",
)
L = st.sidebar.slider(
    "Needle length L (≤ d)",
    min_value=0.1,
    max_value=float(d),
    value=min(1.0, float(d)),
    step=0.1,
    key="buffon_L",
)

st.sidebar.markdown("---")
st.sidebar.header("Randomness")
seed_on = st.sidebar.checkbox(
    "Use fixed seed (repeatable)", value=False, key="buffon_seed_on"
)
seed_val = None
if seed_on:
    seed_val = st.sidebar.number_input(
        "Seed value (Buffon)",
        min_value=0,
        max_value=2147483647,
        value=123,
        step=1,
        key="buffon_seed_val",
    )

# --- Stepwise state management ---
seed_changed = (
    seed_on != st.session_state.buffon_last_seed_on
    or (seed_on and seed_val != st.session_state.buffon_last_seed)
)

if reset_clicked or seed_changed:
    st.session_state.buffon_theta = np.array([])
    st.session_state.buffon_x = np.array([])
    st.session_state.buffon_crosses = np.array([], dtype=bool)
    st.session_state.buffon_n_current = 0

# Ensure RNG is initialized
if seed_changed or (st.session_state.buffon_rng is None):
    if seed_on:
        st.session_state.buffon_rng = np.random.default_rng(seed_val)
        st.session_state.buffon_last_seed_on = seed_on
        st.session_state.buffon_last_seed = seed_val
    else:
        st.session_state.buffon_rng = np.random.default_rng()
        st.session_state.buffon_last_seed_on = seed_on
        st.session_state.buffon_last_seed = None

n_current = int(st.session_state.buffon_n_current)

n_to_add = 0
if next_clicked:
    n_to_add = 1
elif add100_clicked:
    n_to_add = 100
elif finish_clicked:
    n_to_add = max(0, int(n_target) - n_current)

if n_to_add > 0:
    rng = st.session_state.buffon_rng
    theta_new = rng.uniform(0, np.pi / 2, size=n_to_add)
    x_new = rng.uniform(0, d / 2.0, size=n_to_add)
    crosses_new = x_new <= (L / 2.0) * np.sin(theta_new)

    st.session_state.buffon_theta = np.concatenate(
        [st.session_state.buffon_theta, theta_new]
    )
    st.session_state.buffon_x = np.concatenate(
        [st.session_state.buffon_x, x_new]
    )
    st.session_state.buffon_crosses = np.concatenate(
        [st.session_state.buffon_crosses, crosses_new]
    )
    st.session_state.buffon_n_current = n_current + n_to_add

# Clamp current n to available data
n_current = min(
    int(st.session_state.buffon_n_current), len(st.session_state.buffon_crosses)
)

# Compute π estimate from first n_current needles
crosses_use = st.session_state.buffon_crosses[:n_current]
n_cross = int(np.count_nonzero(crosses_use))
p_hat = n_cross / n_current if n_current > 0 else 0.0
pi_est = np.nan if p_hat == 0 else (2.0 * L) / (d * p_hat)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Estimated π",
        f"{pi_est:.6f}" if np.isfinite(pi_est) else "undefined",
    )
with col2:
    st.metric("True π", f"{np.pi:.6f}")
with col3:
    st.metric("Crossing fraction", f"{p_hat:.4f} ({n_cross}/{n_current})")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Needle drops visualization")
    # Wider figure and wider x-range to make the "floor" feel larger horizontally
    fig, ax = plt.subplots(figsize=(8, 4))
    n_lines = 6
    y_lines = np.arange(0, n_lines * d, d)
    for y in y_lines:
        ax.axhline(y, color="black", linewidth=1, alpha=0.4)

    # Use the actual sampled theta and X to place needles consistently with the crossing test.
    theta_use = st.session_state.buffon_theta[:n_current]
    xdist_use = st.session_state.buffon_x[
        :n_current
    ]  # distance to nearest line
    crosses_use = st.session_state.buffon_crosses[:n_current]

    rng_viz = np.random.default_rng(12345)
    # Choose which line is "nearest" for each needle
    line_indices = rng_viz.integers(0, n_lines, size=n_current)
    nearest_lines = y_lines[line_indices]
    # Randomly place center to be above or below that nearest line by the sampled distance X
    signs = rng_viz.choice([-1.0, 1.0], size=n_current)
    centers_y = nearest_lines + signs * xdist_use
    # Spread needle centers across the full visible width
    centers_x = rng_viz.uniform(0, 3 * d, size=n_current)

    for cx, cy, ang, hit in zip(centers_x, centers_y, theta_use, crosses_use):
        dx = (L / 2.0) * np.cos(ang)
        dy = (L / 2.0) * np.sin(ang)
        x0, x1 = cx - dx, cx + dx
        y0, y1 = cy - dy, cy + dy
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=("red" if hit else "blue"),
            linewidth=1,
            alpha=0.8,
        )

    # Show several strips horizontally so the floor feels wide, not narrow
    ax.set_xlim(0, 3 * d)
    ax.set_ylim(0, n_lines * d)
    ax.set_aspect("equal", adjustable="box")
    # Remove axis titles/labels for a cleaner diagram
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col_right:
    st.subheader("Convergence of π estimate")
    if n_current <= 1:
        st.info("Click 'Next step' to start the run and see convergence.")
    else:
        n_vals = choose_n_values(min(n_current, n_target), 50)
        # Running estimates: pi_hat(n) = 2L/(d * (crosses/n))
        crosses_int = st.session_state.buffon_crosses[:n_current].astype(int)
        cumsum = np.cumsum(crosses_int)
        p_running = np.divide(
            cumsum, np.arange(1, n_current + 1), dtype=float
        )
        pi_running = np.where(
            p_running > 0, (2.0 * L) / (d * p_running), np.nan
        )

        pi_vals = pi_running[np.array(n_vals) - 1]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(
            n_vals,
            pi_vals,
            "b-",
            linewidth=1.5,
            alpha=0.7,
            label="π estimate",
        )
        ax.scatter(
            n_vals, pi_vals, s=25, c="blue", alpha=0.6, zorder=5
        )
        ax.axhline(
            np.pi,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"True π = {np.pi:.6f}",
        )
        ax.axvline(
            n_current,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"current n = {n_current:,}",
        )

        ax.set_xscale("log")
        ax.set_xlabel("n (number of needles)")
        ax.set_ylabel("π estimate")
        ax.set_title("Convergence of Buffon π estimate")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

