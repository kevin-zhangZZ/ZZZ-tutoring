"""
Streamlit app for Monte Carlo simulations.

This is the home page for the multipage app. Use the sidebar page
selector to open each simulation.
"""

import streamlit as st

st.set_page_config(
    page_title="Monte Carlo Simulations",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Monte Carlo Simulations")

# Global sidebar styling for a modern, clean look
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

st.markdown(
    "Welcome! This app contains interactive Monte Carlo demos "
    "for teaching and exploring probability and estimation."
)

st.markdown("#### Available simulation pages")
st.markdown(
    "- **01 – Area under curve (sin(πx) on [0,1])**  \n"
    "  Estimate the area under the curve using random points in the unit square.  \n"
    "  This page focuses on Monte Carlo integration, cumulative estimates, and\n"
    "  convergence behavior as the number of samples increases.\n\n"
    "- **02 – Buffon's Needle (π estimation)**  \n"
    "  Drop a virtual needle onto an infinite floor with parallel lines to estimate π\n"
    "  from the fraction of crossings. The page explains the geometric probability\n"
    "  setup and shows how the π estimator converges as more drops are added.\n\n"
    "- **03 – Dice Gift Simulator (2X vs X₁ + X₂)**  \n"
    "  Compare two discrete random variables: doubling a single die roll (2X) versus\n"
    "  summing two independent dice (X₁ + X₂). Side-by-side histograms highlight the\n"
    "  difference between a uniform discrete distribution and a triangular sum\n"
    "  distribution with a peak around 7.\n\n"
    "- **04 – Euler's Method Explorer (Numerical Differential Equations)**  \n"
    "  Visualize Euler's method for approximating solutions to first-order ODEs\n"
    "  y' = f(x,y) with initial conditions. Compare step sizes, observe error growth,\n"
    "  and explore Improved Euler (Heun's method). Includes slope fields and exact\n"
    "  solution comparisons for multiple differential equation models.\n\n"
    "- **05 – First Principles of Calculus (The Derivative)**  \n"
    "  Explore the fundamental definition of the derivative using limits and the\n"
    "  difference quotient. Includes interactive Desmos visualization showing how\n"
    "  secant lines approach the tangent line, plus numerical demonstrations of\n"
    "  convergence for various functions."
)

st.info(
    "Use the **page selector in the sidebar** to switch between simulations. "
    "Each page has its own controls and visualizations."
)

