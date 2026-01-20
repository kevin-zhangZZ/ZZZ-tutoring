# Monte Carlo Area Estimation Demo

An interactive Streamlit application for demonstrating Monte Carlo integration to estimate the area under a curve.

## Overview

This app estimates the area under the curve $y = \sin(\pi x)$ on the interval $[0,1]$ using Monte Carlo simulation. Random points are uniformly sampled in the unit square $[0,1] \times [0,1]$, and the area estimate is computed as the fraction of points that fall under the curve.

**Mathematical Background:**
- **Estimate formula:** $\frac{\text{# points with } y < \sin(\pi x)}{n}$
- **True value:** $\int_0^1 \sin(\pi x) \, dx = \frac{2}{\pi} \approx 0.6366$

## Project Structure

```
.
├── app.py              # Streamlit UI and state management
├── mc_core.py          # Pure numpy functions (no Streamlit dependencies)
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone or download the project** to your local machine.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

Start the Streamlit app with:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## Usage Guide

### Simulation Controls
- **n_display**: Number of points to use for the current estimate and scatter plot (100 to 200,000)
- **Use fixed seed (repeatable)**: Checkbox to enable reproducible results
- **Seed value**: Random seed for reproducibility (only shown when "Use fixed seed" is enabled)
- **Generate / Resample points**: Button to generate new random points

### Display Controls
- **Show random dots**: Toggle to show/hide scatter points in the unit-square plot
- **max_display**: Maximum number of points to render (downsampling for performance, 500 to 50,000)

### Convergence Controls
- **Show convergence plot**: Toggle to show/hide the convergence visualization
- **n_max**: Maximum n value for the convergence curve (1,000 to 500,000)
- **steps**: Number of points/markers on the convergence curve (10 to 300)
- **Auto-update plots when sliders change**: When ON, sliders update plots immediately using stored data. When OFF, changes apply only after clicking "Generate / Resample points"

### Main Display

The app shows:
- **Metrics**: Current estimate, true value (2/π), and error
- **Left plot**: Unit square with the curve $y = \sin(\pi x)$, shaded area under the curve, and scatter points (green = under curve, red = above curve)
- **Right plot**: Convergence visualization showing how the estimate approaches the true value as $n$ increases (log scale on x-axis)

## For Teachers: Educational Notes

### Learning Objectives
1. **Monte Carlo Integration**: Students learn how random sampling can estimate integrals
2. **Convergence**: Students observe that estimates improve (converge) as sample size increases
3. **Statistical Variation**: With fixed seed OFF, students see different results each time
4. **Visualization**: The scatter plot makes the concept concrete—green points "count" toward the area estimate

### Classroom Activities

**Activity 1: Exploration**
- Start with small n (e.g., 100 points)
- Observe the scatter plot and estimate
- Increase n gradually and watch the estimate stabilize
- Compare with the true value (2/π)

**Activity 2: Convergence Investigation**
- Enable the convergence plot
- Adjust n_max and steps to explore different scales
- Discuss: Why does the estimate fluctuate more at small n?
- Why does it converge to 2/π?

**Activity 3: Reproducibility**
- Enable "Use fixed seed"
- Click "Generate / Resample" multiple times—should get the same result
- Disable fixed seed and generate again—results change
- Discuss: When would fixed seeds be useful?

**Activity 4: Performance**
- Set n_display to 50,000 or higher
- Toggle "Show random dots" on/off
- Adjust max_display to see downsampling effect
- Discuss: Why is downsampling necessary for visualization?

### Key Concepts to Emphasize

1. **Law of Large Numbers**: As $n \to \infty$, the sample mean converges to the population mean (here, the true area)
2. **Monte Carlo Method**: Using random sampling to approximate deterministic quantities
3. **Accuracy vs. Computation**: More points = better estimate but slower computation
4. **Randomness**: The role of randomness in numerical methods

### Mathematical Details

- **True Integral**: $\int_0^1 \sin(\pi x) \, dx = \left[-\frac{1}{\pi}\cos(\pi x)\right]_0^1 = \frac{2}{\pi}$
- **Monte Carlo Estimate**: For uniform sampling in $[0,1]^2$, if $p$ is the probability that a point is under the curve, then the area estimate is $p \approx \frac{\text{# hits}}{n}$
- **Expected Value**: The expected value of the estimate is exactly $\frac{2}{\pi}$ (unbiased)

### Assessment Ideas

1. Have students predict what happens when n_display increases
2. Ask students to explain why the convergence plot shows more variation at small n
3. Challenge: Can students derive the true value $\frac{2}{\pi}$?
4. Extension: Modify the curve to estimate a different integral

## Technical Details

### State Management
The app uses Streamlit's `st.session_state` to persist simulation data across reruns:
- Arrays `x_all`, `y_all`, `under_all` store all generated points
- Points are only regenerated when explicitly requested or when more points are needed
- Existing arrays are extended (not regenerated) when `m_required` increases

### Performance Optimizations
- Vectorized numpy operations for fast computation
- Downsampling for visualization (compute uses all points, display uses subset)
- Efficient state management to avoid unnecessary regeneration

## Troubleshooting

**App won't start:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

**Plots not updating:**
- Check that "Auto-update plots when sliders change" is enabled
- Click "Generate / Resample points" to force an update

**Performance issues with large n:**
- Reduce `max_display` to speed up rendering
- Disable "Show random dots" for very large n

## License

This project is provided as-is for educational purposes.
