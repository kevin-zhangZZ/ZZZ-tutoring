"""
First Principles of Calculus: Understanding the derivative from first principles.

This page explains the fundamental concept of the derivative using the limit definition
and provides interactive visualizations to help students understand the geometric
and algebraic meaning of differentiation.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components


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


def init_calculus_state() -> None:
    """Initialize session state for calculus demonstrations."""
    if "calc_results" not in st.session_state:
        st.session_state.calc_results = None


# Initialize state
init_calculus_state()

st.title("First Principles of Calculus: The Derivative")

st.markdown(
    "This page explores the **fundamental definition of the derivative** using first principles. "
    "We'll see how the derivative emerges naturally from the concept of a limit and understand "
    "its geometric meaning as the slope of a tangent line."
)

st.markdown("#### What is the derivative?")
st.markdown(
    "The derivative of a function $f(x)$ at a point $x = a$ measures the **instantaneous rate of change** "
    "of the function at that point. Geometrically, it represents the **slope of the tangent line** "
    "to the curve $y = f(x)$ at the point $(a, f(a))$."
)

st.markdown("#### The limit definition (first principles)")
st.markdown(
    "The derivative is defined as the limit of the difference quotient as the interval shrinks to zero:"
)
st.latex(r"f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}")

st.markdown(
    "This formula comes from taking the slope of a **secant line** through points $(a, f(a))$ and "
    "$(a+h, f(a+h))$, then letting $h$ approach zero. As $h \\to 0$, the secant line becomes the "
    "**tangent line** at $x = a$."
)

st.markdown("#### Geometric interpretation")
st.markdown(
    "- **Secant line**: A line through two points on the curve. Its slope is the **average rate of change** "
    "over the interval.\n"
    "- **Tangent line**: The limiting position of the secant line as the two points approach each other. "
    "Its slope is the **instantaneous rate of change** (the derivative).\n"
    "- As $h$ gets smaller, the secant line rotates and approaches the tangent line."
)

st.markdown("#### Alternative notation")
st.markdown(
    "The derivative can also be written using the limit definition with $x$ approaching $a$:"
)
st.latex(r"f'(a) = \lim_{x \to a} \frac{f(x) - f(a)}{x - a}")

st.markdown("#### Interactive visualization")
st.markdown(
    "Use the Desmos graph below to explore how the secant line approaches the tangent line as $h$ approaches zero. "
    "You can adjust the point $a$ and the value of $h$ to see the geometric relationship."
)

# Embed Desmos graph
st.markdown("---")
st.subheader("Interactive Graph: Secant to Tangent")
st.markdown(
    "Use the controls in the Desmos graph below to adjust the point $a$ and the value of $h$ "
    "to see how the secant line approaches the tangent line. "
    "**Tip:** Click on the graph to interact with it, or use the sliders if visible."
)

# Provide link to open in Desmos directly
st.markdown(
    "[**Open in Desmos**](https://www.desmos.com/calculator/mu5ozejroc) (opens in new tab for full interactivity)"
)

# Use Streamlit components for proper iframe embedding
# Note: Some browsers may restrict iframe interactions - use the link above if needed
components.iframe(
    "https://www.desmos.com/calculator/mu5ozejroc?embed",
    width=900,
    height=650,
    scrolling=False,
)

st.markdown("---")

# Sidebar controls for numerical demonstration
st.sidebar.header("Numerical Demonstration")
st.sidebar.markdown(
    "Choose a function and point to see how the difference quotient approaches the derivative."
)

function_choice = st.sidebar.selectbox(
    "Choose a function f(x)",
    [
        "f(x) = x²",
        "f(x) = x³",
        "f(x) = sin(x)",
        "f(x) = e^x",
        "f(x) = ln(x)",
    ],
    key="calc_function",
)

# Define functions and their derivatives
FUNCTIONS = {
    "f(x) = x²": {
        "f": lambda x: x**2,
        "f_prime": lambda x: 2 * x,
        "domain": (-5, 5),
    },
    "f(x) = x³": {
        "f": lambda x: x**3,
        "f_prime": lambda x: 3 * x**2,
        "domain": (-3, 3),
    },
    "f(x) = sin(x)": {
        "f": lambda x: np.sin(x),
        "f_prime": lambda x: np.cos(x),
        "domain": (-2 * np.pi, 2 * np.pi),
    },
    "f(x) = e^x": {
        "f": lambda x: np.exp(x),
        "f_prime": lambda x: np.exp(x),
        "domain": (-2, 2),
    },
    "f(x) = ln(x)": {
        "f": lambda x: np.log(x),
        "f_prime": lambda x: 1 / x,
        "domain": (0.1, 5),
    },
}

func_info = FUNCTIONS[function_choice]
a = st.sidebar.number_input(
    "Point a (where to compute derivative)",
    value=1.0,
    step=0.1,
    key="calc_point_a",
    min_value=float(func_info["domain"][0]) + 0.1,
    max_value=float(func_info["domain"][1]) - 0.1,
)

h_values = st.sidebar.multiselect(
    "Values of h to compare",
    [0.5, 0.1, 0.05, 0.01, 0.001],
    default=[0.5, 0.1, 0.01],
    key="calc_h_values",
    help="Select multiple h values to see how the difference quotient approaches the derivative",
)

show_tangent = st.sidebar.checkbox(
    "Show tangent line",
    value=True,
    key="calc_show_tangent",
)

# Compute and display results
if len(h_values) > 0:
    f = func_info["f"]
    f_prime = func_info["f_prime"]
    
    # True derivative value
    true_derivative = f_prime(a)
    f_a = f(a)
    
    # Compute difference quotients
    diff_quotients = {}
    for h in h_values:
        diff_quot = (f(a + h) - f(a)) / h
        diff_quotients[h] = diff_quot
    
    st.session_state.calc_results = {
        "function": function_choice,
        "a": a,
        "f_a": f_a,
        "true_derivative": true_derivative,
        "diff_quotients": diff_quotients,
        "h_values": h_values,
    }

if st.session_state.calc_results is not None:
    res = st.session_state.calc_results
    
    # Display numerical results
    st.subheader("Numerical Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("f(a)", f"{res['f_a']:.6f}")
        st.metric("True derivative f'(a)", f"{res['true_derivative']:.6f}")
    
    with col2:
        st.markdown("**Difference quotients:**")
        for h in sorted(res["h_values"], reverse=True):
            dq = res["diff_quotients"][h]
            error = abs(dq - res["true_derivative"])
            st.markdown(f"$h = {h}$: $\\frac{{f(a+h) - f(a)}}{{h}} = {dq:.6f}$")
            st.caption(f"Error: {error:.6e}")
    
    # Visual demonstration
    st.subheader("Visual Demonstration")
    
    func_info = FUNCTIONS[res["function"]]
    f = func_info["f"]
    f_prime = func_info["f_prime"]
    domain = func_info["domain"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the function
    x_plot = np.linspace(domain[0], domain[1], 1000)
    y_plot = f(x_plot)
    ax.plot(x_plot, y_plot, "b-", linewidth=2, label=f"${res['function']}$")
    
    # Mark the point (a, f(a))
    ax.plot(res["a"], res["f_a"], "ro", markersize=10, label=f"Point $(a, f(a)) = ({res['a']:.2f}, {res['f_a']:.2f})$")
    
    # Plot secant lines for different h values
    colors = plt.cm.viridis(np.linspace(0, 1, len(res["h_values"])))
    for i, h in enumerate(sorted(res["h_values"], reverse=True)):
        x_sec = np.array([res["a"] - 1, res["a"] + h + 1])
        slope = res["diff_quotients"][h]
        y_sec = res["f_a"] + slope * (x_sec - res["a"])
        ax.plot(
            x_sec,
            y_sec,
            "--",
            color=colors[i],
            linewidth=1.5,
            alpha=0.7,
            label=f"Secant (h={h}, slope={slope:.4f})",
        )
        # Mark the second point
        ax.plot(res["a"] + h, f(res["a"] + h), "o", color=colors[i], markersize=6, alpha=0.7)
    
    # Plot tangent line
    if show_tangent:
        x_tangent = np.linspace(res["a"] - 1.5, res["a"] + 1.5, 100)
        y_tangent = res["f_a"] + res["true_derivative"] * (x_tangent - res["a"])
        ax.plot(
            x_tangent,
            y_tangent,
            "k-",
            linewidth=2.5,
            alpha=0.8,
            label=f"Tangent (slope={res['true_derivative']:.4f})",
        )
    
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(
        f"First Principles: Secant Lines Approaching Tangent\n"
        f"{res['function']} at $x = {res['a']:.2f}$",
        fontsize=13,
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Zoom in around the point of interest
    ax.set_xlim(res["a"] - 2, res["a"] + 2)
    y_range = max([abs(f(res["a"] + h) - res["f_a"]) for h in res["h_values"]]) * 2
    ax.set_ylim(res["f_a"] - y_range, res["f_a"] + y_range)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Explanation
    st.markdown("#### Observations")
    st.markdown(
        "- As $h$ gets smaller, the secant line's slope approaches the tangent line's slope.\n"
        "- The difference quotient $\\frac{f(a+h) - f(a)}{h}$ gets closer to the true derivative $f'(a)$.\n"
        "- The error decreases as $h \\to 0$, demonstrating the limit process."
    )

else:
    st.info("Select values of $h$ in the sidebar to see the numerical demonstration.")

st.markdown("---")

st.markdown("#### Example: Finding the derivative of $f(x) = x^2$")
st.markdown("Using first principles:")
st.latex(
    r"""
    \begin{align*}
    f'(x) &= \lim_{h \to 0} \frac{(x+h)^2 - x^2}{h} \\
          &= \lim_{h \to 0} \frac{x^2 + 2xh + h^2 - x^2}{h} \\
          &= \lim_{h \to 0} \frac{2xh + h^2}{h} \\
          &= \lim_{h \to 0} (2x + h) \\
          &= 2x
    \end{align*}
    """
)

st.markdown("#### Key takeaways")
st.markdown(
    "- The derivative is defined as a **limit**, not just a formula.\n"
    "- First principles show us **why** the derivative works, not just **how** to compute it.\n"
    "- The geometric interpretation (slope of tangent) connects algebra to visual understanding.\n"
    "- This foundation is essential for understanding more advanced calculus concepts."
)

st.markdown("---")
st.markdown(
    "*Use the interactive Desmos graph and numerical demonstration above to explore "
    "how the secant line becomes the tangent line as $h \\to 0$.*"
)
