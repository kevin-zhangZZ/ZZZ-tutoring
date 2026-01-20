"""
Unit Circle Teaching & Testing Module for VCE Specialist Maths.

Learn, Practice, and Quiz on:
- Special unit circle angles
- Exact values of sin, cos, tan
- Angle ↔ point relationships
"""

import time
import random
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import sympy as sp


# -----------------------------
# Shared styling for sidebar
# -----------------------------

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


# -----------------------------
# Angle data / constants
# -----------------------------

# Angles in radians as SymPy expressions (0 to 11π/6)
ANGLE_SYMPY: List[sp.Expr] = [
    sp.Integer(0),
    sp.pi / 6,
    sp.pi / 4,
    sp.pi / 3,
    sp.pi / 2,
    2 * sp.pi / 3,
    3 * sp.pi / 4,
    5 * sp.pi / 6,
    sp.pi,
    7 * sp.pi / 6,
    5 * sp.pi / 4,
    4 * sp.pi / 3,
    3 * sp.pi / 2,
    5 * sp.pi / 3,
    7 * sp.pi / 4,
    11 * sp.pi / 6,
]

# LaTeX display for those angles
ANGLE_LATEX: List[str] = [
    "0",
    r"\frac{\pi}{6}",
    r"\frac{\pi}{4}",
    r"\frac{\pi}{3}",
    r"\frac{\pi}{2}",
    r"\frac{2\pi}{3}",
    r"\frac{3\pi}{4}",
    r"\frac{5\pi}{6}",
    r"\pi",
    r"\frac{7\pi}{6}",
    r"\frac{5\pi}{4}",
    r"\frac{4\pi}{3}",
    r"\frac{3\pi}{2}",
    r"\frac{5\pi}{3}",
    r"\frac{7\pi}{4}",
    r"\frac{11\pi}{6}",
]

# Degrees for each angle
ANGLE_DEGREES: List[int] = [
    0,
    30,
    45,
    60,
    90,
    120,
    135,
    150,
    180,
    210,
    225,
    240,
    270,
    300,
    315,
    330,
]

# Labels 1..16 for points around the unit circle
ANGLE_LABELS: List[str] = [str(i + 1) for i in range(len(ANGLE_SYMPY))]

N_ANGLES = len(ANGLE_SYMPY)


# -----------------------------
# Helper functions
# -----------------------------

def get_angle_sympy(idx: int) -> sp.Expr:
    """Return the SymPy angle for a given index (0-based)."""
    return ANGLE_SYMPY[idx % N_ANGLES]


def exact_trig(angle: sp.Expr, fn: str) -> Tuple[str, bool, sp.Expr]:
    """
    Compute exact trig value for special angles using SymPy.

    Returns (latex_str, is_undefined, expr)
    """
    fn = fn.lower()
    if fn == "sin":
        expr = sp.simplify(sp.sin(angle))
    elif fn == "cos":
        expr = sp.simplify(sp.cos(angle))
    elif fn == "tan":
        cos_val = sp.simplify(sp.cos(angle))
        if sp.simplify(cos_val) == 0:
            return r"\text{Undefined}", True, sp.nan
        expr = sp.simplify(sp.tan(angle))
    else:
        raise ValueError(f"Unknown trig function '{fn}'")

    return sp.latex(expr), False, expr


def normalize_user_answer(text: str) -> str:
    """Basic cleaning for user text prior to parsing."""
    if text is None:
        return ""
    s = text.strip()
    # Remove enclosing $...$ or spaces
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    # Common replacements
    replacements = {
        "π": "pi",
        "√": "sqrt",
        "^": "**",
        "−": "-",  # unicode minus
        " ": "",
        "\\left": "",
        "\\right": "",
        "\\cdot": "*",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    # Normalize degree symbol
    s = s.replace("deg", "°")
    return s


def parse_trig_answer(text: str) -> Union[sp.Expr, str, None]:
    """
    Parse user trig answer.

    Returns:
        sympy.Expr for numeric values,
        'undefined' for undefined/DNE,
        None if parsing fails.
    """
    s = normalize_user_answer(text).lower()
    if not s:
        return None

    # Handle undefined / DNE
    if s in {"undefined", "undef", "dne", "doesnotexist", "does_not_exist", "na"}:
        return "undefined"

    try:
        expr = sp.sympify(s)
        return sp.simplify(expr)
    except Exception:
        return None


def check_trig_answer(
    user_text: str, correct_expr: sp.Expr, is_undefined: bool
) -> bool:
    """Compare user trig answer against the exact value."""
    parsed = parse_trig_answer(user_text)
    if parsed is None:
        return False

    if is_undefined:
        return isinstance(parsed, str) and parsed == "undefined"

    if isinstance(parsed, str) and parsed == "undefined":
        return False

    try:
        diff = sp.simplify(parsed - correct_expr)
        return diff == 0
    except Exception:
        return False


def parse_angle_answer(
    text: str, allow_degrees: bool, coterminal: bool
) -> Optional[sp.Expr]:
    """
    Parse user angle answer; supports radians and (optionally) degrees.
    """
    s = normalize_user_answer(text)
    if not s:
        return None

    # Degrees
    if allow_degrees and ("°" in s or s.endswith("deg") or s.endswith("degree")):
        # remove non-numeric markers
        s_deg = (
            s.replace("°", "")
            .replace("deg", "")
            .replace("degree", "")
            .replace("degrees", "")
        )
        try:
            deg_val = float(s_deg)
        except ValueError:
            return None
        angle = sp.nsimplify(deg_val * sp.pi / 180)
    elif allow_degrees:
        # Could be plain number meaning degrees
        try:
            deg_val = float(s)
            angle = sp.nsimplify(deg_val * sp.pi / 180)
        except Exception:
            try:
                angle = sp.nsimplify(s)
            except Exception:
                return None
    else:
        try:
            angle = sp.nsimplify(s)
        except Exception:
            return None

    if coterminal:
        # Map to principal range [0, 2π)
        two_pi = 2 * sp.pi
        angle = sp.simplify(angle % two_pi)
    return angle


def check_angle_answer(
    user_text: str,
    correct_angle: sp.Expr,
    allow_degrees: bool,
    coterminal: bool,
) -> bool:
    """Compare user angle answer against correct angle (mod 2π if enabled)."""
    user_angle = parse_angle_answer(user_text, allow_degrees, coterminal)
    if user_angle is None:
        return False

    if coterminal:
        two_pi = 2 * sp.pi
        diff = sp.simplify((user_angle - correct_angle) % two_pi)
        return diff == 0
    diff = sp.simplify(user_angle - correct_angle)
    return diff == 0


def unit_circle_coordinates(angle: sp.Expr) -> Tuple[sp.Expr, sp.Expr]:
    """Return exact (cosθ, sinθ) for an angle."""
    x = sp.simplify(sp.cos(angle))
    y = sp.simplify(sp.sin(angle))
    return x, y


def angle_quadrant(angle: sp.Expr) -> str:
    """Determine quadrant I–IV or axis for a given angle (0 ≤ θ < 2π)."""
    two_pi = 2 * sp.pi
    theta = sp.simplify(angle % two_pi)
    if theta == 0 or theta == sp.pi or theta == sp.pi / 2 or theta == 3 * sp.pi / 2:
        return "axis"
    if 0 < theta < sp.pi / 2:
        return "I"
    if sp.pi / 2 < theta < sp.pi:
        return "II"
    if sp.pi < theta < 3 * sp.pi / 2:
        return "III"
    return "IV"


def reference_angle(angle: sp.Expr) -> sp.Expr:
    """Compute reference angle in (0, π/2] for a given angle."""
    two_pi = 2 * sp.pi
    theta = sp.simplify(angle % two_pi)
    if 0 <= theta <= sp.pi / 2:
        return theta
    if sp.pi / 2 <= theta <= sp.pi:
        return sp.pi - theta
    if sp.pi <= theta <= 3 * sp.pi / 2:
        return theta - sp.pi
    return 2 * sp.pi - theta


# -----------------------------
# Plotting: unit circle
# -----------------------------

def plot_unit_circle(
    highlight_idx: Optional[int] = None,
    show_radian_labels: bool = False,
    show_reference_triangle: bool = False,
) -> plt.Figure:
    """Create a unit circle plot with optional highlight and reference triangle."""
    fig, ax = plt.subplots(figsize=(4, 4))

    # Axes lines
    ax.axhline(0, color="#e0e0e0", lw=1.5)
    ax.axvline(0, color="#e0e0e0", lw=1.5)

    # Circle
    pastel_color = "#90caf9"
    circle = plt.Circle((0, 0), 1, fill=False, color=pastel_color, lw=3)
    ax.add_artist(circle)

    point_color = "#00897b"
    label_color = "#004d40"

    for i, angle in enumerate(ANGLE_SYMPY):
        x_val = float(sp.cos(angle))
        y_val = float(sp.sin(angle))
        if highlight_idx is not None and i == highlight_idx:
            ax.plot(
                x_val,
                y_val,
                "o",
                color="#ff7043",
                markersize=14,
                markeredgecolor="black",
                markeredgewidth=1.2,
            )
        else:
            ax.plot(x_val, y_val, "o", color=point_color, markersize=10)

        if show_radian_labels:
            label_text = ANGLE_LATEX[i].replace(r"\pi", "π")
        else:
            label_text = ANGLE_LABELS[i]

        ax.text(
            x_val * 1.25,
            y_val * 1.25,
            label_text,
            color=label_color,
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
        )

    # Optional reference triangle for highlighted point
    if show_reference_triangle and highlight_idx is not None:
        angle = ANGLE_SYMPY[highlight_idx]
        x_exact, y_exact = unit_circle_coordinates(angle)
        x = float(x_exact)
        y = float(y_exact)
        ax.plot([0, x], [0, y], color="#3949ab", lw=2)  # radius
        ax.plot([x, x], [0, y], color="#b0bec5", lw=2, linestyle="--")
        ax.plot([0, x], [0, 0], color="#b0bec5", lw=2, linestyle="--")
        ax.text(
            x / 2,
            -0.07,
            sp.latex(sp.simplify(sp.Abs(x_exact))),
            ha="center",
            va="top",
            fontsize=11,
        )
        ax.text(
            x + 0.08,
            y / 2,
            sp.latex(sp.simplify(sp.Abs(y_exact))),
            ha="left",
            va="center",
            fontsize=11,
        )

    ax.set_xlim([-1.35, 1.35])
    ax.set_ylim([-1.35, 1.35])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig


# -----------------------------
# Streamlit page layout
# -----------------------------

st.title("Unit Circle Explorer – Special Angles & Exact Trig Values")

st.markdown(
    "This mini-module helps you learn and practise the **unit circle**, "
    "special angles, and exact values of $\\sin$, $\\cos$, and $\\tan$ – "
    "aligned with VCE Specialist Maths."
)

tab_learn, tab_practice, tab_quiz = st.tabs(["Learn", "Practice", "Quiz"])


# -----------------------------
# LEARN TAB
# -----------------------------

with tab_learn:
    colL, colR = st.columns([1, 1])

    with colL:
        st.subheader("Unit Circle Reference")
        idx_learn = st.selectbox(
            "Choose a point on the circle",
            options=list(range(1, N_ANGLES + 1)),
            index=0,
            key="uc_learn_point",
        )
        angle_idx = idx_learn - 1

        show_radian_labels = st.checkbox(
            "Show radians as labels", value=False, key="uc_learn_show_rad_labels"
        )
        show_triangle = st.checkbox(
            "Show reference triangle", value=True, key="uc_learn_ref_triangle"
        )

        fig_uc = plot_unit_circle(
            highlight_idx=angle_idx,
            show_radian_labels=show_radian_labels,
            show_reference_triangle=show_triangle,
        )
        st.pyplot(fig_uc)

    with colR:
        st.subheader("Angle Information")
        angle = get_angle_sympy(angle_idx)
        deg = ANGLE_DEGREES[angle_idx]
        cos_ltx, _, cos_expr = exact_trig(angle, "cos")
        sin_ltx, _, sin_expr = exact_trig(angle, "sin")
        tan_ltx, tan_undef, tan_expr = exact_trig(angle, "tan")

        st.markdown("**Exact angle values:**")
        st.latex(
            rf"\theta = {ANGLE_LATEX[angle_idx]} \quad\text{{({deg}^\circ)}}"
        )

        st.markdown("**Coordinates on the unit circle:**")
        st.latex(
            rf"(x,y) = (\cos\theta,\sin\theta) = \left({cos_ltx},\,{sin_ltx}\right)"
        )

        st.markdown("**Exact trig values:**")
        st.latex(rf"\sin\theta = {sin_ltx}")
        st.latex(rf"\cos\theta = {cos_ltx}")
        if tan_undef:
            st.latex(r"\tan\theta \text{ is undefined (vertical tangent)}")
        else:
            st.latex(rf"\tan\theta = {tan_ltx}")

        quad = angle_quadrant(angle)
        ref = reference_angle(angle)
        st.markdown("**Quadrant & reference angle:**")
        if quad == "axis":
            st.markdown(
                "- This angle lies on one of the axes, so the reference angle is the angle to that axis."
            )
        else:
            st.markdown(
                rf"- Quadrant: **{quad}**  \n"
                rf"- Reference angle: ${sp.latex(ref)}$"
            )

        st.markdown("**ASTC rule (signs of trig ratios):**")
        st.markdown(
            "- Quadrant I: All trig ratios positive (A).  \n"
            "- Quadrant II: Sine positive, cos & tan negative (S).  \n"
            "- Quadrant III: Tangent positive, sin & cos negative (T).  \n"
            "- Quadrant IV: Cosine positive, sin & tan negative (C)."
        )


# -----------------------------
# PRACTICE TAB
# -----------------------------

QUESTION_TYPES = [
    "label→angle (radians)",
    "label→degrees",
    "angle→label",
    "exact sin",
    "exact cos",
    "exact tan",
    "coordinates (cosθ, sinθ)",
]


def init_practice_state() -> None:
    if "uc_practice" not in st.session_state:
        st.session_state.uc_practice = {
            "current_type": None,
            "angle_idx": None,
            "question_text": "",
            "answer_latex": "",
            "correct_trig_expr": None,
            "correct_angle": None,
            "is_undefined": False,
            "hint_level": 0,
        }


def generate_practice_question(selected_types: List[str]) -> None:
    state = st.session_state.uc_practice
    if not selected_types:
        return
    q_type = random.choice(selected_types)
    idx = random.randint(0, N_ANGLES - 1)
    theta = get_angle_sympy(idx)
    deg = ANGLE_DEGREES[idx]

    state["current_type"] = q_type
    state["angle_idx"] = idx
    state["hint_level"] = 0
    state["correct_trig_expr"] = None
    state["correct_angle"] = None
    state["is_undefined"] = False

    if q_type == "label→angle (radians)":
        state["question_text"] = (
            f"Which angle (in radians) does point **{ANGLE_LABELS[idx]}** represent?"
        )
        state["answer_latex"] = ANGLE_LATEX[idx]
        state["correct_angle"] = theta
    elif q_type == "label→degrees":
        state["question_text"] = (
            f"Which angle (in degrees) does point **{ANGLE_LABELS[idx]}** represent?"
        )
        state["answer_latex"] = rf"{deg}^\circ"
        state["correct_angle"] = theta
    elif q_type == "angle→label":
        state["question_text"] = (
            rf"Which point label corresponds to the angle $\theta = {ANGLE_LATEX[idx]}$?"
        )
        state["answer_latex"] = ANGLE_LABELS[idx]
    elif q_type in {"exact sin", "exact cos", "exact tan"}:
        fn = q_type.split()[-1]  # sin, cos, tan
        state["question_text"] = (
            rf"Find the exact value of ${fn}({ANGLE_LATEX[idx]})$."
        )
        ltx, is_undef, expr = exact_trig(theta, fn)
        state["answer_latex"] = ltx
        state["is_undefined"] = is_undef
        state["correct_trig_expr"] = expr
    elif q_type == "coordinates (cosθ, sinθ)":
        state["question_text"] = (
            rf"Find the exact coordinates $(\cos\theta,\sin\theta)$ for "
            rf"$\theta = {ANGLE_LATEX[idx]}$."
        )
        x_expr, y_expr = unit_circle_coordinates(theta)
        state["answer_latex"] = (
            r"\left(" + sp.latex(x_expr) + r",\, " + sp.latex(y_expr) + r"\right)"
        )
        state["correct_trig_expr"] = (x_expr, y_expr)


with tab_practice:
    init_practice_state()
    state = st.session_state.uc_practice

    st.subheader("Guided Practice")
    st.markdown(
        "Choose the types of questions you want to practise, then use **Next question**. "
        "Enter your answer in exact form (fractions and square roots)."
    )

    col_cfg, col_q = st.columns([1, 2])

    with col_cfg:
        st.markdown("**Question types**")
        type_checks = {}
        for qt in QUESTION_TYPES:
            default_on = qt in {
                "label→angle (radians)",
                "exact sin",
                "exact cos",
                "exact tan",
            }
            type_checks[qt] = st.checkbox(
                qt, value=default_on, key=f"uc_practice_type_{qt}"
            )
        selected_types = [qt for qt, on in type_checks.items() if on]

        allow_deg = st.checkbox(
            "Allow degrees in angle answers",
            value=True,
            key="uc_practice_allow_deg",
        )
        show_hint_btn = st.checkbox(
            "Show hint button", value=True, key="uc_practice_show_hint"
        )
        show_reveal_btn = st.checkbox(
            "Show reveal solution button",
            value=True,
            key="uc_practice_show_reveal",
        )

        show_location = st.checkbox(
            "Show angle location on unit circle",
            value=True,
            key="uc_practice_show_location",
        )

        if st.button("Next question", key="uc_practice_next"):
            generate_practice_question(selected_types)

    with col_q:
        if state["current_type"] is None:
            st.info("Click **Next question** to begin.")
        else:
            idx = state["angle_idx"]
            theta = get_angle_sympy(idx)
            st.write(state["question_text"])

            # Always show the unit circle; optionally highlight the current point
            highlight = idx if st.session_state.get("uc_practice_show_location", True) else None
            fig_small = plot_unit_circle(
                highlight_idx=highlight,
                show_radian_labels=False,
                show_reference_triangle=False,
            )
            st.pyplot(fig_small)

            user_answer = st.text_input(
                "Your answer (use fractions / sqrt where needed):",
                key="uc_practice_answer",
            )

            # Hints
            if show_hint_btn:
                if st.button("Show hint", key="uc_practice_hint"):
                    state["hint_level"] = min(3, state["hint_level"] + 1)

            if state["hint_level"] > 0:
                quad = angle_quadrant(theta)
                st.markdown(f"**Hint 1 – Quadrant & sign (ASTC):**")
                if quad == "axis":
                    st.markdown(
                        "- This angle lies on an axis; think about which coordinate is 0 or ±1."
                    )
                else:
                    st.markdown(
                        rf"- Quadrant **{quad}** – use ASTC to decide which trig ratios are positive."
                    )
            if state["hint_level"] > 1:
                ref = reference_angle(theta)
                st.markdown("**Hint 2 – Reference angle:**")
                st.markdown(
                    rf"- Reference angle is ${sp.latex(ref)}$; compare with the first-quadrant special angle."
                )
            if state["hint_level"] > 2:
                st.markdown("**Hint 3 – Base special-angle values:**")
                st.markdown(
                    "- The basic magnitudes you should know are "
                    r"$\frac{1}{2}, \frac{\sqrt{2}}{2}, \frac{\sqrt{3}}{2}$."
                )

            # Check answer
            if user_answer.strip():
                correct = False
                explanation_lines: List[str] = []

                if state["current_type"] in {
                    "exact sin",
                    "exact cos",
                    "exact tan",
                }:
                    correct_expr = state["correct_trig_expr"]
                    correct = check_trig_answer(
                        user_answer, correct_expr, state["is_undefined"]
                    )
                    explanation_lines.append(
                        "We use the reference angle and quadrant to determine the exact trig value."
                    )
                elif state["current_type"] in {
                    "label→angle (radians)",
                    "label→degrees",
                }:
                    correct_angle = state["correct_angle"]
                    correct = check_angle_answer(
                        user_answer,
                        correct_angle,
                        allow_degrees=allow_deg
                        or state["current_type"] == "label→degrees",
                        coterminal=True,
                    )
                    explanation_lines.append(
                        "Angles that differ by a multiple of $2\\pi$ are coterminal and represent the same point."
                    )
                elif state["current_type"] == "angle→label":
                    norm = normalize_user_answer(user_answer)
                    correct = norm == ANGLE_LABELS[idx]
                    explanation_lines.append(
                        "Each label corresponds to a fixed point on the circle; match the angle to its position."
                    )
                elif state["current_type"] == "coordinates (cosθ, sinθ)":
                    # Expect something like "(a,b)" or "a,b"
                    s = normalize_user_answer(user_answer)
                    if s.startswith("(") and s.endswith(")"):
                        s = s[1:-1]
                    if "," in s:
                        part_x, part_y = s.split(",", 1)
                        parsed_x = parse_trig_answer(part_x)
                        parsed_y = parse_trig_answer(part_y)
                        x_expr, y_expr = state["correct_trig_expr"]
                        if (
                            isinstance(parsed_x, sp.Expr)
                            and isinstance(parsed_y, sp.Expr)
                        ):
                            ok_x = sp.simplify(parsed_x - x_expr) == 0
                            ok_y = sp.simplify(parsed_y - y_expr) == 0
                            correct = ok_x and ok_y
                    explanation_lines.append(
                        "Use $(x,y) = (\\cos\\theta, \\sin\\theta)$ and your knowledge of special triangles."
                    )

                if correct:
                    st.success("✅ Correct!")
                else:
                    st.error("❌ Not quite.")

                st.markdown("**Exact answer:**")
                st.latex(state["answer_latex"])

                if explanation_lines:
                    st.markdown("**Explanation:**")
                    for line in explanation_lines:
                        st.markdown(f"- {line}")

            if show_reveal_btn and st.button(
                "Reveal solution", key="uc_practice_reveal"
            ):
                st.markdown("**Exact answer:**")
                st.latex(state["answer_latex"])


# -----------------------------
# QUIZ TAB
# -----------------------------

def init_quiz_state() -> None:
    if "uc_quiz" not in st.session_state:
        st.session_state.uc_quiz = {
            "active": False,
            "start_time": None,
            "time_limit": None,
            "n_questions": 10,
            "current_index": 0,
            "score": 0,
            "records": [],
            "current_question": None,
        }


def generate_quiz_angle(
    base_idx: int, difficulty: str
) -> Tuple[sp.Expr, str, int, int]:
    """
    Generate an angle for quiz based on difficulty.

    Returns (angle_expr, latex_display, degrees, base_idx)
    """
    base_angle = ANGLE_SYMPY[base_idx]
    deg = ANGLE_DEGREES[base_idx]
    angle_expr = base_angle
    latex_display = ANGLE_LATEX[base_idx]

    if difficulty == "Core 16 angles only":
        return angle_expr, latex_display, deg, base_idx

    k = random.choice([-1, 0, 1, 2])  # shift multiplier for 2π

    if difficulty == "Include negative angles":
        if k >= 0:
            k = -1
        angle_expr = base_angle + k * 2 * sp.pi
    elif difficulty == "Include angles > 2π":
        if k <= 0:
            k = random.choice([1, 2])
        angle_expr = base_angle + k * 2 * sp.pi
    elif difficulty == "Include coterminal angle recognition":
        angle_expr = base_angle + k * 2 * sp.pi

    # LaTeX display
    if k == 0:
        latex_display = ANGLE_LATEX[base_idx]
    else:
        sign = "+" if k > 0 else "-"
        k_abs = abs(k)
        if k_abs == 1:
            extra = r"2\pi"
        else:
            extra = rf"{k_abs}\cdot 2\pi"
        latex_display = ANGLE_LATEX[base_idx] + f" {sign} " + extra

    return angle_expr, latex_display, deg, base_idx


with tab_quiz:
    init_quiz_state()
    qstate = st.session_state.uc_quiz

    st.subheader("Quiz Mode")
    st.markdown(
        "Test yourself with a short quiz on unit circle angles and exact trig values. "
        "Answers should be in exact form where possible."
    )

    if not qstate["active"]:
        # Settings before starting
        n_questions = st.slider(
            "Number of questions",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
            key="uc_quiz_n_questions",
        )
        time_limit_on = st.checkbox(
            "Use time limit", value=False, key="uc_quiz_time_limit_on"
        )
        time_limit = None
        if time_limit_on:
            time_limit = st.slider(
                "Time limit (seconds)",
                min_value=60,
                max_value=300,
                value=180,
                step=30,
                key="uc_quiz_time_limit_value",
            )
        difficulty = st.selectbox(
            "Difficulty",
            [
                "Core 16 angles only",
                "Include negative angles",
                "Include angles > 2π",
                "Include coterminal angle recognition",
            ],
            key="uc_quiz_difficulty",
        )
        categories = st.multiselect(
            "Question types included",
            options=QUESTION_TYPES,
            default=["exact sin", "exact cos", "exact tan", "label→angle (radians)"],
            key="uc_quiz_categories",
        )
        allow_deg_quiz = st.checkbox(
            "Allow degrees in angle answers",
            value=True,
            key="uc_quiz_allow_deg",
        )

        if st.button("Start quiz", key="uc_quiz_start"):
            qstate["active"] = True
            qstate["start_time"] = time.time()
            qstate["time_limit"] = time_limit
            qstate["n_questions"] = n_questions
            qstate["current_index"] = 0
            qstate["score"] = 0
            qstate["records"] = []
            qstate["current_question"] = {
                "difficulty": difficulty,
                "categories": categories,
                "allow_deg": allow_deg_quiz,
            }
    else:
        # Active quiz
        elapsed = time.time() - qstate["start_time"]
        if qstate["time_limit"] is not None and elapsed > qstate["time_limit"]:
            st.warning("⏰ Time is up! Finishing the quiz.")
            qstate["active"] = False

        if not qstate["active"]:
            # Fall through to summary rendering below
            pass
        elif qstate["current_index"] >= qstate["n_questions"]:
            qstate["active"] = False
        else:
            # Generate current question if needed
            meta = qstate["current_question"]
            diff = meta["difficulty"]
            cats = meta["categories"]
            allow_deg_quiz = meta["allow_deg"]

            if "working_question" not in qstate:
                # New question
                q_type = random.choice(cats)
                base_idx = random.randint(0, N_ANGLES - 1)
                theta_quiz, theta_latex, deg, base_idx = generate_quiz_angle(
                    base_idx, diff
                )

                record = {
                    "q_number": qstate["current_index"] + 1,
                    "question_type": q_type,
                    "base_idx": base_idx,
                    "display_angle": theta_latex,
                    "your_answer": "",
                    "correct_answer": "",
                    "is_correct": False,
                    "quadrant": angle_quadrant(get_angle_sympy(base_idx)),
                    "ref_angle": sp.latex(reference_angle(get_angle_sympy(base_idx))),
                }

                if q_type in {"exact sin", "exact cos", "exact tan"}:
                    fn = q_type.split()[-1]
                    ltx, is_undef, expr = exact_trig(theta_quiz, fn)
                    record["correct_answer"] = ltx
                    q_current = {
                        "text": rf"Find the exact value of ${fn}({theta_latex})$.",
                        "kind": "trig",
                        "expr": expr,
                        "is_undefined": is_undef,
                        "record": record,
                    }
                elif q_type in {"label→angle (radians)", "label→degrees"}:
                    record["correct_answer"] = (
                        ANGLE_LATEX[base_idx]
                        if q_type == "label→angle (radians)"
                        else rf"{deg}^\circ"
                    )
                    q_current = {
                        "text": f"Which angle does point **{ANGLE_LABELS[base_idx]}** represent?",
                        "kind": "angle",
                        "angle": get_angle_sympy(base_idx),
                        "record": record,
                        "allow_deg": allow_deg_quiz
                        or q_type == "label→degrees",
                    }
                else:
                    # Fallback to trig-style question
                    fn = "sin"
                    ltx, is_undef, expr = exact_trig(theta_quiz, fn)
                    record["correct_answer"] = ltx
                    q_current = {
                        "text": rf"Find the exact value of $\sin({theta_latex})$.",
                        "kind": "trig",
                        "expr": expr,
                        "is_undefined": is_undef,
                        "record": record,
                    }

                qstate["working_question"] = q_current

            q = qstate["working_question"]

            st.markdown(
                f"**Question {qstate['current_index'] + 1} of {qstate['n_questions']}**"
            )
            if qstate["time_limit"] is not None:
                remaining = max(0, int(qstate["time_limit"] - elapsed))
                st.caption(f"Time remaining: {remaining} seconds")

            st.write(q["text"])
            user_ans = st.text_input(
                "Your answer:",
                key=f"uc_quiz_answer_{qstate['current_index']}",
            )

            if st.button("Submit answer", key=f"uc_quiz_submit_{qstate['current_index']}"):
                correct = False
                rec = q["record"]
                rec["your_answer"] = user_ans

                if q["kind"] == "trig":
                    correct = check_trig_answer(
                        user_ans, q["expr"], q["is_undefined"]
                    )
                elif q["kind"] == "angle":
                    correct = check_angle_answer(
                        user_ans,
                        q["angle"],
                        allow_degrees=q.get("allow_deg", True),
                        coterminal=True,
                    )

                rec["is_correct"] = bool(correct)
                qstate["records"].append(rec)
                if correct:
                    qstate["score"] += 1
                    st.success("✅ Correct!")
                else:
                    st.error("❌ Not quite.")
                    st.markdown("**Exact answer:**")
                    st.latex(rec["correct_answer"])

                # Advance
                qstate["current_index"] += 1
                del qstate["working_question"]

    # Summary if quiz inactive and we have records
    if not qstate["active"] and qstate["records"]:
        st.subheader("Quiz Summary")
        n = len(qstate["records"])
        score = sum(1 for r in qstate["records"] if r["is_correct"])
        accuracy = 100 * score / n if n > 0 else 0
        st.markdown(f"**Score:** {score} / {n}  \n**Accuracy:** {accuracy:.1f}%")

        if qstate["time_limit"] is not None:
            elapsed_total = int(elapsed)
            avg_time = elapsed_total / max(1, n)
            st.caption(
                f"Total time: {elapsed_total} seconds  •  Average time per question: {avg_time:.1f} s"
            )

        # Review table of missed questions
        missed = [r for r in qstate["records"] if not r["is_correct"]]
        if missed:
            st.markdown("#### Review missed questions")
            headers = [
                "Q#",
                "Type",
                "Angle (displayed)",
                "Your answer",
                "Correct answer",
                "Quadrant",
                "Reference angle",
            ]
            rows = []
            for r in missed:
                rows.append(
                    [
                        r["q_number"],
                        r["question_type"],
                        r["display_angle"],
                        r["your_answer"],
                        r["correct_answer"],
                        r["quadrant"],
                        r["ref_angle"],
                    ]
                )
            st.table(rows)
        else:
            st.success("Perfect score – no missed questions to review!")

