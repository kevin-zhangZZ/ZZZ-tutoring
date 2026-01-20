"""
Dice Gift Simulator: comparing 2X vs X1 + X2 distributions.

This page illustrates the difference between multiplying a single die roll by 2
versus summing two independent die rolls.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


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


def init_dice_state() -> None:
    """Initialize session state for dice simulations."""
    if "dice_scenario1_results" not in st.session_state:
        st.session_state.dice_scenario1_results = None
    if "dice_scenario2_results" not in st.session_state:
        st.session_state.dice_scenario2_results = None


# Initialize state for this page
init_dice_state()

st.title("Dice Gift Simulator: 2X vs X₁ + X₂")

st.markdown(
    "This page demonstrates an important distinction in probability theory: "
    "the difference between **multiplying a single random variable by 2** "
    "versus **summing two independent random variables**."
)

st.markdown("#### Setup")
st.markdown(
    "Imagine you receive a gift based on dice rolls. We compare two scenarios:"
)
st.markdown(
    "- **Scenario 1 (2X):** Roll one fair die, multiply the result by 2.\n"
    "- **Scenario 2 (X₁ + X₂):** Roll two fair dice independently, sum the results."
)

st.markdown("#### Mathematical framework")
st.markdown("Let $X$ be a fair die roll, so $X \\sim \\mathrm{Uniform}\\{1,2,3,4,5,6\\}$.")

st.markdown("**Scenario 1: $Y = 2X$**")
st.latex(r"Y = 2X \quad \text{where } X \sim \mathrm{Uniform}\{1,2,3,4,5,6\}")
st.markdown(
    "Since $X$ takes values in $\\{1,2,3,4,5,6\\}$, the random variable $Y = 2X$ "
    "takes values in $\\{2,4,6,8,10,12\\}$ with equal probability $\\frac{1}{6}$ each. "
    "The distribution is **discrete and uniform** over these even numbers."
)

st.markdown("**Scenario 2: $Z = X_1 + X_2$**")
st.latex(
    r"Z = X_1 + X_2 \quad \text{where } X_1, X_2 \sim \mathrm{Uniform}\{1,2,3,4,5,6\} \text{ (independent)}"
)
st.markdown(
    "Here $Z$ takes values in $\\{2,3,4,\\ldots,12\\}$, but the probabilities are **not uniform**. "
    "The sum of two independent uniform discrete random variables follows a triangular distribution:"
)
st.latex(r"P(Z = k) = \frac{6 - |k-7|}{36} \quad \text{for } k \in \{2,3,\ldots,12\}")
st.markdown(
    "This peaks at $Z = 7$ (probability $\\frac{6}{36} = \\frac{1}{6}$) and decreases symmetrically "
    "toward the extremes. For example, $P(Z=2) = P(Z=12) = \\frac{1}{36}$."
)

st.markdown("#### Key insight")
st.markdown(
    "Even though both scenarios can produce values in $\\{2,4,6,8,10,12\\}$ (for Scenario 1) "
    "or $\\{2,\\ldots,12\\}$ (for Scenario 2), the **distributions are fundamentally different**:"
)
st.markdown(
    "- **$2X$**: Only even outcomes, uniform probability.\n"
    "- **$X_1 + X_2$**: All integer outcomes 2–12, triangular (peaked) distribution."
)

st.markdown("---")

# Two-column layout for simulations
col1, col2 = st.columns(2)

# ===== COLUMN 1: Scenario 1 (2X) =====
with col1:
    st.subheader("Scenario 1: 1 die × 2")
    
    runs1 = st.number_input(
        "Number of simulations (Scenario 1):",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        key="dice_runs1",
        help="Number of times to roll one die and multiply by 2",
    )
    
    sim1_clicked = st.button("Simulate Scenario 1", type="primary", key="dice_sim1")
    
    if sim1_clicked or st.session_state.dice_scenario1_results is not None:
        if sim1_clicked:
            # Generate new simulation
            rng = np.random.default_rng()
            rolls = rng.integers(1, 7, size=int(runs1))
            gifts = rolls * 2
            st.session_state.dice_scenario1_results = {
                "gifts": gifts,
                "runs": int(runs1),
            }
        
        res1 = st.session_state.dice_scenario1_results
        if res1 is not None:
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Histogram with bins aligned to even outcomes 2,4,6,8,10,12
            bins = np.arange(2, 14, 2) - 0.5  # [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5]
            ax.hist(
                res1["gifts"],
                bins=bins,
                edgecolor="black",
                alpha=0.7,
                color="steelblue",
            )
            
            ax.set_xticks([2, 4, 6, 8, 10, 12])
            ax.set_xlabel("Gift Amount", fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.set_title(
                f"Scenario 1: 2X (n={res1['runs']:,})",
                fontsize=12,
            )
            ax.grid(True, alpha=0.3, axis="y")
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Show summary statistics
            unique, counts = np.unique(res1["gifts"], return_counts=True)
            st.caption(f"**Observed outcomes:** {', '.join(map(str, unique))}")
            st.caption(
                f"**Expected frequency per outcome:** {res1['runs']/6:.1f} "
                f"(theoretical: {res1['runs']/6:.1f})"
            )

# ===== COLUMN 2: Scenario 2 (X1 + X2) =====
with col2:
    st.subheader("Scenario 2: 2 dice sum")
    
    runs2 = st.number_input(
        "Number of simulations (Scenario 2):",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        key="dice_runs2",
        help="Number of times to roll two dice and sum them",
    )
    
    sim2_clicked = st.button("Simulate Scenario 2", type="primary", key="dice_sim2")
    
    if sim2_clicked or st.session_state.dice_scenario2_results is not None:
        if sim2_clicked:
            # Generate new simulation
            rng = np.random.default_rng()
            rolls = rng.integers(1, 7, size=(int(runs2), 2))
            gifts = np.sum(rolls, axis=1)
            st.session_state.dice_scenario2_results = {
                "gifts": gifts,
                "runs": int(runs2),
            }
        
        res2 = st.session_state.dice_scenario2_results
        if res2 is not None:
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Histogram with bins aligned to integer outcomes 2..12
            bins = np.arange(2, 14) - 0.5  # [1.5, 2.5, ..., 12.5]
            ax.hist(
                res2["gifts"],
                bins=bins,
                edgecolor="black",
                alpha=0.7,
                color="coral",
            )
            
            ax.set_xticks(np.arange(2, 13))
            ax.set_xlabel("Gift Amount", fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.set_title(
                f"Scenario 2: X₁ + X₂ (n={res2['runs']:,})",
                fontsize=12,
            )
            ax.grid(True, alpha=0.3, axis="y")
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Show summary statistics
            unique, counts = np.unique(res2["gifts"], return_counts=True)
            max_count_idx = np.argmax(counts)
            st.caption(f"**Observed outcomes:** {', '.join(map(str, unique))}")
            st.caption(
                f"**Most frequent outcome:** {unique[max_count_idx]} "
                f"(observed {counts[max_count_idx]} times, "
                f"theoretical peak at 7 with probability 1/6)"
            )

st.markdown("---")
st.markdown(
    "*Run both simulations and compare the histograms to see the difference "
    "between $2X$ and $X_1 + X_2$ distributions.*"
)
