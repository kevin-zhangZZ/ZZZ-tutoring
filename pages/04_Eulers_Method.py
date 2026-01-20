"""
Euler's Method Explorer: Numerical solution of differential equations.

This page demonstrates Euler's method for approximating solutions to
first-order ODEs y' = f(x,y) with initial conditions.
Designed for VCE Specialist Maths students.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd


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


def init_euler_state() -> None:
    """Initialize session state for Euler's method simulations."""
    if "euler_results" not in st.session_state:
        st.session_state.euler_results = None


# Initialize state
init_euler_state()

st.title("Euler's Method Explorer (Numerical Differential Equations)")

st.markdown(
    "This interactive tool demonstrates **Euler's method**, a numerical technique "
    "for approximating solutions to first-order differential equations of the form "
    "$y' = f(x, y)$ with an initial condition $y(x_0) = y_0$."
)

st.markdown("#### What problem does Euler's method solve?")
st.markdown(
    "Many differential equations don't have closed-form solutions that we can write down. "
    "Euler's method gives us a way to **approximate** the solution curve by stepping forward "
    "from the initial point $(x_0, y_0)$ using the slope information from the differential equation."
)

st.markdown("#### The Euler update rule")
st.latex(r"y_{n+1} = y_n + h \cdot f(x_n, y_n)")
st.markdown(
    "where $h$ is the **step size** and $f(x_n, y_n)$ is the slope at point $(x_n, y_n)$ "
    "given by the differential equation. We also update $x$:"
)
st.latex(r"x_{n+1} = x_n + h")

st.markdown("#### Understanding step size $h$")
st.markdown(
    "- **Smaller $h$**: More accurate approximation (more steps, slower computation)\n"
    "- **Larger $h$**: Less accurate approximation (fewer steps, faster computation)\n"
    "- The error typically grows as we step further from the initial condition."
)

st.markdown("#### Try this")
st.markdown(
    "1. Compare $h=1$ vs $h=0.1$ for $y' = y$ with $y(0)=1$ — observe how accuracy improves with smaller steps.\n"
    "2. Watch the error grow over time — Euler's method accumulates error.\n"
    "3. Try the **Improved Euler (Heun's method)** toggle — it uses a better slope estimate and is more accurate."
)

st.markdown("---")

# Define ODE functions and their exact solutions
ODE_FUNCTIONS = {
    "y' = y (exponential growth)": {
        "f": lambda x, y: y,
        "exact": lambda x, x0, y0: y0 * np.exp(x - x0),
        "description": "Exponential growth model",
    },
    "y' = -y (exponential decay)": {
        "f": lambda x, y: -y,
        "exact": lambda x, x0, y0: y0 * np.exp(-(x - x0)),
        "description": "Exponential decay model",
    },
    "y' = x (simple integration)": {
        "f": lambda x, y: x,
        "exact": lambda x, x0, y0: y0 + (x**2 - x0**2) / 2,
        "description": "Integrates to a parabola",
    },
    "y' = x + y (linear ODE)": {
        "f": lambda x, y: x + y,
        "exact": lambda x, x0, y0: (y0 + x0 + 1) * np.exp(x - x0) - x - 1,
        "description": "Linear first-order ODE",
    },
    "y' = sin(x) (independent of y)": {
        "f": lambda x, y: np.sin(x),
        "exact": lambda x, x0, y0: y0 - np.cos(x) + np.cos(x0),
        "description": "Right-hand side depends only on x",
    },
}

# Sidebar controls
st.sidebar.header("Differential Equation")
ode_choice = st.sidebar.selectbox(
    "Choose a differential equation y' = f(x, y)",
    list(ODE_FUNCTIONS.keys()),
    key="euler_ode_choice",
)
ode_info = ODE_FUNCTIONS[ode_choice]
st.sidebar.caption(f"*{ode_info['description']}*")

st.sidebar.header("Initial Conditions")
x0 = st.sidebar.number_input(
    "x₀ (start x)",
    value=0.0,
    step=0.1,
    key="euler_x0",
)
y0 = st.sidebar.number_input(
    "y₀ (initial value)",
    value=1.0,
    step=0.1,
    key="euler_y0",
)
xf = st.sidebar.number_input(
    "x_f (end x)",
    value=5.0,
    min_value=x0 + 0.01,
    step=0.1,
    key="euler_xf",
)

st.sidebar.header("Step Size")
h_single = st.sidebar.slider(
    "Step size h (single run)",
    min_value=0.01,
    max_value=1.0,
    value=0.1,
    step=0.01,
    key="euler_h_single",
    help="Step size for a single Euler approximation",
)

h_presets = st.sidebar.multiselect(
    "Compare multiple step sizes",
    [0.05, 0.1, 0.25, 0.5, 1.0],
    default=[0.1, 0.5],
    key="euler_h_multiselect",
    help="Select multiple h values to compare on the same plot",
)

st.sidebar.header("Method")
use_improved_euler = st.sidebar.checkbox(
    "Use Improved Euler (Heun's method)",
    value=False,
    key="euler_improved",
    help="Uses a better slope estimate: average of slopes at start and predicted end of step",
)

st.sidebar.header("Display Options")
show_exact = st.sidebar.checkbox(
    "Show exact solution (if available)",
    value=True,
    key="euler_show_exact",
)
show_error = st.sidebar.checkbox(
    "Show error graph",
    value=False,
    key="euler_show_error",
)
show_slope_field = st.sidebar.checkbox(
    "Show slope field",
    value=False,
    key="euler_show_slope_field",
)

# Euler's method implementation
def euler_method(f, x0, y0, xf, h, improved=False):
    """
    Solve y' = f(x,y) using Euler's method or Improved Euler (Heun's).
    
    Parameters:
    -----------
    f : callable f(x, y)
        Right-hand side of ODE
    x0, y0 : float
        Initial condition
    xf : float
        Final x value
    h : float
        Step size
    improved : bool
        If True, use Improved Euler (Heun's method)
    
    Returns:
    --------
    x : ndarray
        Array of x values
    y : ndarray
        Array of y values (approximations)
    """
    x_vals = [x0]
    y_vals = [y0]
    
    x = x0
    y = y0
    
    while x < xf:
        if improved:
            # Improved Euler (Heun's method)
            k1 = f(x, y)
            y_predict = y + h * k1
            k2 = f(x + h, y_predict)
            y = y + (h / 2) * (k1 + k2)
        else:
            # Standard Euler
            y = y + h * f(x, y)
        
        x = x + h
        x_vals.append(x)
        y_vals.append(y)
    
    return np.array(x_vals), np.array(y_vals)


# Run simulations
run_clicked = st.button("Run Euler's Method", type="primary", key="euler_run")

if run_clicked or st.session_state.euler_results is not None:
    f = ode_info["f"]
    exact_func = ode_info.get("exact")
    
    # Determine which h values to use
    h_values = h_presets if len(h_presets) > 0 else [h_single]
    
    results = {}
    
    for h in h_values:
        x_arr, y_arr = euler_method(f, x0, y0, xf, h, improved=use_improved_euler)
        results[h] = {"x": x_arr, "y": y_arr}
    
    st.session_state.euler_results = {
        "results": results,
        "ode_choice": ode_choice,
        "x0": x0,
        "y0": y0,
        "xf": xf,
        "exact_func": exact_func,
        "improved": use_improved_euler,
    }

# Plotting
if st.session_state.euler_results is not None:
    res = st.session_state.euler_results
    results = res["results"]
    exact_func = res["exact_func"]
    
    # Main plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Euler approximations
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for i, (h, data) in enumerate(results.items()):
        method_label = "Improved Euler" if res["improved"] else "Euler"
        ax.plot(
            data["x"],
            data["y"],
            "o-",
            label=f"{method_label} (h={h})",
            color=colors[i],
            markersize=4,
            linewidth=1.5,
            alpha=0.8,
        )
    
    # Plot exact solution if available and enabled
    if exact_func is not None and show_exact:
        x_exact = np.linspace(x0, xf, 1000)
        y_exact = exact_func(x_exact, x0, y0)
        ax.plot(
            x_exact,
            y_exact,
            "k--",
            label="Exact solution",
            linewidth=2,
            alpha=0.7,
        )
    
    # Slope field
    if show_slope_field:
        # Determine y range from solutions
        y_min = min([np.min(data["y"]) for data in results.values()])
        y_max = max([np.max(data["y"]) for data in results.values()])
        if exact_func is not None:
            x_test = np.linspace(x0, xf, 50)
            y_test = exact_func(x_test, x0, y0)
            y_min = min(y_min, np.min(y_test))
            y_max = max(y_max, np.max(y_test))
        
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
        
        # Create grid for slope field
        x_grid = np.linspace(x0, xf, 15)
        y_grid = np.linspace(y_min, y_max, 15)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Compute slopes
        dx_sf = (xf - x0) / 20
        dy_sf = dx_sf * f(X_grid, Y_grid)
        
        # Normalize arrow lengths for visibility
        norm = np.sqrt(dx_sf**2 + dy_sf**2)
        scale = 0.8 * dx_sf / (norm + 1e-10)
        dx_sf = dx_sf * scale
        dy_sf = dy_sf * scale
        
        ax.quiver(
            X_grid,
            Y_grid,
            dx_sf,
            dy_sf,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="gray",
            alpha=0.4,
            width=0.002,
        )
    
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    method_name = "Improved Euler (Heun)" if res["improved"] else "Euler"
    ax.set_title(
        f"{method_name} Method: {res['ode_choice']}\n"
        f"Initial condition: y({x0}) = {y0}",
        fontsize=13,
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Error plot
    if show_error and exact_func is not None and len(results) > 0:
        fig_err, ax_err = plt.subplots(figsize=(10, 4))
        
        for i, (h, data) in enumerate(results.items()):
            y_exact_at_points = exact_func(data["x"], x0, y0)
            errors = np.abs(y_exact_at_points - data["y"])
            ax_err.plot(
                data["x"],
                errors,
                "o-",
                label=f"Error (h={h})",
                color=colors[i],
                markersize=3,
                linewidth=1.5,
                alpha=0.8,
            )
        
        ax_err.set_xlabel("x", fontsize=12)
        ax_err.set_ylabel("Absolute error |y_exact - y_approx|", fontsize=11)
        ax_err.set_title("Error Growth Over Time", fontsize=12)
        ax_err.legend(loc="best", fontsize=9)
        ax_err.grid(True, alpha=0.3)
        ax_err.set_yscale("log")
        
        plt.tight_layout()
        st.pyplot(fig_err)
        plt.close(fig_err)
    
    # Results table
    st.subheader("Numerical Results")
    
    # Use smallest h for the table, or first h if only one
    h_table = min(results.keys()) if len(results) > 0 else h_single
    if h_table in results:
        data_table = results[h_table]
        x_table = data_table["x"]
        y_table = data_table["y"]
        
        # Limit to first 30 rows
        n_show = min(30, len(x_table))
        x_table = x_table[:n_show]
        y_table = y_table[:n_show]
        
        df_data = {"n": np.arange(n_show), "x_n": x_table, "y_n": y_table}
        
        if exact_func is not None:
            y_exact_table = exact_func(x_table, x0, y0)
            df_data["y_exact"] = y_exact_table
            df_data["error"] = np.abs(y_exact_table - y_table)
        
        df = pd.DataFrame(df_data)
        
        # Format columns
        for col in ["x_n", "y_n"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.6f}")
        if "y_exact" in df.columns:
            df["y_exact"] = df["y_exact"].apply(lambda x: f"{x:.6f}")
        if "error" in df.columns:
            df["error"] = df["error"].apply(lambda x: f"{x:.6e}")
        
        st.dataframe(df, use_container_width=True)
        
        if len(x_table) < len(data_table["x"]):
            st.caption(f"*Table shows first {n_show} rows. Total steps: {len(data_table['x'])}*")
        else:
            st.caption(f"*Total steps: {len(data_table['x'])}*")
        
        st.caption(f"*Results shown for h = {h_table}*")

else:
    st.info("Configure the parameters above and click 'Run Euler's Method' to see the approximation.")

st.markdown("---")
st.markdown(
    "*Adjust step sizes, compare methods, and observe how numerical approximations "
    "compare to exact solutions (when available).*"
)
