import io
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Polygon


st.set_page_config(page_title="Super-T4 bending app", layout="wide")




# -----------------------------
# Embedded fallback libraries
# -----------------------------
DEFAULT_SUPERT = {
    "Super_T1": {
        "b_f_top_mm": 500,   # top flange width each side from inner gap
        "h_f_top_mm": 90,     # top flange thickness

        "b_f_bot_mm": 899,    # bottom width
        "h_f_bot_mm": 245,    # bottom thickness at centre soffit reference
        "b_w_mm": 100,        # web-related thickness parameter
        "H_mm": 765,         # total depth

        "gap_top1_mm": 843,
        "gap_top2_mm": 1027,
        "gap_bot3_mm": 756,
        "incl_mm": 71,},
    
    "Super_T2": {
        "b_f_top_mm": 500,   # top flange width each side from inner gap
        "h_f_top_mm": 90,     # top flange thickness

        "b_f_bot_mm": 852,    # bottom width
        "h_f_bot_mm": 245,    # bottom thickness at centre soffit reference
        "b_w_mm": 100,        # web-related thickness parameter
        "H_mm": 1015,         # total depth

        "gap_top1_mm": 843,
        "gap_top2_mm": 1027,
        "gap_bot3_mm": 710,
        "incl_mm": 67,},
        
    "Super_T3": {
        "b_f_top_mm": 500,   # top flange width each side from inner gap
        "h_f_top_mm": 90,     # top flange thickness

        "b_f_bot_mm": 814,    # bottom width
        "h_f_bot_mm": 265,    # bottom thickness at centre soffit reference
        "b_w_mm": 100,        # web-related thickness parameter
        "H_mm": 1215,         # total depth

        "gap_top1_mm": 843,
        "gap_top2_mm": 1027,
        "gap_bot3_mm": 676,
        "incl_mm": 64},
  
    "Super_T4": {
        "b_f_top_mm": 500,   # top flange width each side from inner gap
        "h_f_top_mm": 90,     # top flange thickness

        "b_f_bot_mm": 757,    # bottom width
        "h_f_bot_mm": 265,    # bottom thickness at centre soffit reference
        "b_w_mm": 100,        # web-related thickness parameter
        "H_mm": 1515,         # total depth

        "gap_top1_mm": 843,
        "gap_top2_mm": 1027,
        "gap_bot3_mm": 618,
        "incl_mm": 59},
    
    "Super_T5": {
        "b_f_top_mm": 500,   # top flange width each side from inner gap
        "h_f_top_mm": 90,     # top flange thickness

        "b_f_bot_mm": 700,    # bottom width
        "h_f_bot_mm": 330,    # bottom thickness at centre soffit reference
        "b_w_mm": 120,        # web-related thickness parameter
        "H_mm": 1815,         # total depth

        "gap_top1_mm": 803,
        "gap_top2_mm": 1027,
        "gap_bot3_mm": 530,
        "incl_mm": 50},

        "notes": "T4 Super-T beam dimensions based on sectionproperties input"
}

DEFAULT_CONCRETE = {
        "Gr25": {"E_MPa": 27500,
            "nu": 0.2,
            "density_kg_m3": 2400,
            "f'c_MPa": 25,
            "f'ct.f_MPa": 3,
            "fct_MPa": 1.8,
            "notes": "AS 5100.5",
            "fcd_MPa": 13.5}
            ,
        "Gr32": {"E_MPa": 30800,
            "nu": 0.2,
            "density_kg_m3": 2400,
            "f'c_MPa": 32,
            "fcd_MPa": 17.28,
            "f'ct.f_MPa": 3.4,
            "fct_MPa": 2.04,
            "notes": "AS 5100.5"}
            ,
        "Gr40": {"E_MPa": 34100,
            "nu": 0.2,
            "density_kg_m3": 2400,
            "f'c_MPa": 40,
            "fcd_MPa": 21.6,
            "f'ct.f_MPa": 3.8,
            "fct_MPa": 2.28,
            "notes": "AS 5100.5"}
            ,
        "Gr50": {"E_MPa": 35700,
            "nu": 0.2,
            "density_kg_m3": 2400,
            "f'c_MPa": 50,
            "fcd_MPa": 27,
            "f'ct.f_MPa": 4.2,
            "fct_MPa": 2.52,
            "notes": "AS 5100.5"}
            ,
        "Gr65": {"E_MPa": 38500,
            "nu": 0.2,
            "density_kg_m3": 2400,
            "f'c_MPa": 65,
            "fcd_MPa": 35.1,
            "f'ct.f_MPa": 4.8,
            "fct_MPa": 2.88,
            "notes": "AS 5100.5"}
            ,
	"meta": {"standard": "AS 5100.5 (Bridge design - Concrete)",
            "standard_strength_grades_MPa": [25, 32, 40, 50, 65, 80, 100],
            "f'c_MPa": "- characteristic  compressive  (cylinder)  strength  of  concrete  at  28  days, t3.1.2",
            "fcd_MPa": "- principal compressive stresses limiting values for Cl 2.3.3",
            "f'ct_MPa": "- characteristic  uniaxial  tensile  strength,tensile capacity before cracking, = fct_MPa",
            "f'ct.f_MPa": "- characteristic  flexural  tensile  strength  of  concrete  at  28  days, bending -microcracking and stress redistribution make this value higher ",
            "fct_MPa": "- uniaxial  tensile  strength  of  concrete",
    },
}

DEFAULT_STEEL = {
 "Prestress_1860": {
        "type": "prestressing_steel",
        "E_MPa": 195_000,
        "nu": 0.3,
        "density_kg_m3": 7850,
        "f_pb_MPa": 1860,          # characteristic tensile strength
        "f_py_MPa": 0.85 * 1860,   # proof / yield strength (≈ 0.85 fpu)
        "eps_pu": 0.035,           # ultimate strain (typical)
        "relaxation": "low",
        "notes": "7-wire low-relaxation strands; AS/NZS 4672"
    },

    "Prestress_1770": {
        "type": "prestressing_steel",
        "E_MPa": 195_000,
        "nu": 0.3,
        "density_kg_m3": 7850,
        "f_pb_MPa": 1770,
        "f_py_MPa": 0.85 * 1770,
        "eps_pu": 0.035,
        "relaxation": "low",
        "notes": "Prestressing wire/strand; AS/NZS 4672"
    },

    "meta": {
        "standard": [
            "AS 5100.5 – Bridge Design (Concrete) Table 3.2.1",
            "AS/NZS 4671 – Steel reinforcing materials",
            "AS/NZS 4672 – Prestressing steel"
        ],
        "E_steel_MPa": "Elastic modulus of reinforcing steel",
        "f_sy_MPa": "Characteristic yield strength of reinforcing steel, ",
        "f_su_MPa": "Ultimate tensile strength of reinforcing steel",
        "f_pb_MPa": "Characteristic tensile strength of prestressing steel",
        "f_py_MPa": "0.2% proof stress or effective yield of prestressing steel",
        "eps_y": "Yield strain = f_sy / E",
        "eps_pu": "Ultimate strain of prestressing steel",
    },
}

SuperT_PS_library = {
    "Super_T1": [
        {"row": 1, "num_strands": 10, "strand_area": 143, "bottom_distance": 60},
        {"row": 2, "num_strands": 14, "strand_area": 143, "bottom_distance": 110},
        {"row": 3, "num_strands": 14, "strand_area": 143, "bottom_distance": 160},
        {"row": 4, "num_strands": 8,  "strand_area": 143, "bottom_distance": 210},
        {"row": 5, "num_strands": 0,  "strand_area": 0,   "bottom_distance": 260},
        {"row": 6, "num_strands": 2,  "strand_area": 143, "bottom_distance": 685},
    ],
    "Super_T2": [
        {"row": 1, "num_strands": 10, "strand_area": 143, "bottom_distance": 60},
        {"row": 2, "num_strands": 14, "strand_area": 143, "bottom_distance": 110},
        {"row": 3, "num_strands": 12, "strand_area": 143, "bottom_distance": 160},
        {"row": 4, "num_strands": 8,  "strand_area": 143, "bottom_distance": 210},
        {"row": 5, "num_strands": 0,  "strand_area": 0,   "bottom_distance": 260},
        {"row": 6, "num_strands": 2,  "strand_area": 143, "bottom_distance": 935},
    ],
    "Super_T3": [
        {"row": 1, "num_strands": 8,  "strand_area": 143, "bottom_distance": 60},
        {"row": 2, "num_strands": 12, "strand_area": 143, "bottom_distance": 110},
        {"row": 3, "num_strands": 12, "strand_area": 143, "bottom_distance": 160},
        {"row": 4, "num_strands": 12, "strand_area": 143, "bottom_distance": 210},
        {"row": 5, "num_strands": 2,  "strand_area": 143, "bottom_distance": 260},
        {"row": 6, "num_strands": 2,  "strand_area": 143, "bottom_distance": 1135},
    ],
    "Super_T4": [
        {"row": 1, "num_strands": 8,  "strand_area": 143, "bottom_distance": 60},
        {"row": 2, "num_strands": 12, "strand_area": 143, "bottom_distance": 110},
        {"row": 3, "num_strands": 12, "strand_area": 143, "bottom_distance": 160},
        {"row": 4, "num_strands": 12, "strand_area": 143, "bottom_distance": 210},
        {"row": 5, "num_strands": 2,  "strand_area": 143, "bottom_distance": 260},
        {"row": 6, "num_strands": 2,  "strand_area": 143, "bottom_distance": 1435},
    ],
    "Super_T5": [
        {"row": 1, "num_strands": 8,  "strand_area": 143, "bottom_distance": 60},
        {"row": 2, "num_strands": 10, "strand_area": 143, "bottom_distance": 110},
        {"row": 3, "num_strands": 10, "strand_area": 143, "bottom_distance": 160},
        {"row": 4, "num_strands": 12, "strand_area": 143, "bottom_distance": 210},
        {"row": 5, "num_strands": 12, "strand_area": 143, "bottom_distance": 260},
        {"row": 6, "num_strands": 2,  "strand_area": 143, "bottom_distance": 310},
        {"row": 7, "num_strands": 2,  "strand_area": 143, "bottom_distance": 1735},
    ],
}

DEFAULT_PS = SuperT_PS_library

# -----------------------------
# JSON helpers
# -----------------------------
def load_uploaded_json(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        return json.loads(content)
    except Exception as exc:
        st.error(f"Could not parse {uploaded_file.name}: {exc}")
        return None


def strip_meta_keys(d: Dict) -> Dict:
    return {k: v for k, v in d.items() if isinstance(v, dict) and k != "meta"}


# -----------------------------
# Geometry helpers
# -----------------------------
@dataclass
class PolyProps:
    area: float
    cx: float
    cy: float
    ixx_centroid: float
    iyy_centroid: float


def polygon_properties(points: List[Tuple[float, float]]) -> PolyProps:
    pts = points[:]
    if pts[0] != pts[-1]:
        pts.append(pts[0])

    A2 = 0.0
    Cx_num = 0.0
    Cy_num = 0.0
    Ixx_o = 0.0
    Iyy_o = 0.0

    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        cross = x0 * y1 - x1 * y0
        A2 += cross
        Cx_num += (x0 + x1) * cross
        Cy_num += (y0 + y1) * cross
        Ixx_o += (y0**2 + y0 * y1 + y1**2) * cross
        Iyy_o += (x0**2 + x0 * x1 + x1**2) * cross

    area_signed = A2 / 2.0
    if abs(area_signed) < 1e-9:
        raise ValueError("Polygon area is zero.")

    cx = Cx_num / (6.0 * area_signed)
    cy = Cy_num / (6.0 * area_signed)
    Ixx_o /= 12.0
    Iyy_o /= 12.0

    area = abs(area_signed)
    Ixx_o = abs(Ixx_o)
    Iyy_o = abs(Iyy_o)
    ixx_c = Ixx_o - area * cy**2
    iyy_c = Iyy_o - area * cx**2
    return PolyProps(area=area, cx=cx, cy=cy, ixx_centroid=ixx_c, iyy_centroid=iyy_c)


def rectangle_properties(width: float, height: float, x0: float, y0: float) -> PolyProps:
    area = width * height
    cx = x0 + width / 2.0
    cy = y0 + height / 2.0
    ixx = width * height**3 / 12.0
    iyy = height * width**3 / 12.0
    return PolyProps(area=area, cx=cx, cy=cy, ixx_centroid=ixx, iyy_centroid=iyy)


def combine_properties(parts: List[PolyProps]) -> PolyProps:
    area = sum(p.area for p in parts)
    cx = sum(p.area * p.cx for p in parts) / area
    cy = sum(p.area * p.cy for p in parts) / area
    ixx = sum(p.ixx_centroid + p.area * (p.cy - cy) ** 2 for p in parts)
    iyy = sum(p.iyy_centroid + p.area * (p.cx - cx) ** 2 for p in parts)
    return PolyProps(area=area, cx=cx, cy=cy, ixx_centroid=ixx, iyy_centroid=iyy)


def build_super_t_points(d: Dict[str, float]) -> List[Tuple[float, float]]:
    b_f_top = d["b_f_top_mm"]
    h_f_top = d["h_f_top_mm"]
    b_f_bot = d["b_f_bot_mm"]
    h_f_bot = d["h_f_bot_mm"]
    H = d["H_mm"]
    gap_top1 = d["gap_top1_mm"]
    gap_top2 = d["gap_top2_mm"]
    gap_bot3 = d["gap_bot3_mm"]
    incl = d["incl_mm"]

    return [
        (-gap_top1 / 2, H),
        (-(gap_top1 / 2 + b_f_top), H),
        (-(gap_top1 / 2 + b_f_top), H - h_f_top),
        (-(gap_top2 / 2 + h_f_top), H - h_f_top),
        (-gap_top2 / 2, H - 2 * h_f_top),
        (-b_f_bot / 2, 0),
        (b_f_bot / 2, 0),
        (gap_top2 / 2, H - 2 * h_f_top),
        ((gap_top2 / 2 + h_f_top), H - h_f_top),
        ((gap_top1 / 2 + b_f_top), H - h_f_top),
        ((gap_top1 / 2 + b_f_top), H),
        (gap_top1 / 2, H),
        (gap_bot3 / 2, h_f_bot + incl),
        (0, h_f_bot),
        (-gap_bot3 / 2, h_f_bot + incl),
    ]


def width_composite_super_t(y_top: float, beam: Dict[str, float], b_slab: float, t_slab: float) -> float:
    H_beam = beam["H_mm"]
    b_f_top = beam["b_f_top_mm"]
    h_f_top = beam["h_f_top_mm"]
    b_f_bot = beam["b_f_bot_mm"]
    h_f_bot = beam["h_f_bot_mm"]
    b_w = beam["b_w_mm"]
    gap_top1 = beam["gap_top1_mm"]
    gap_top2 = beam["gap_top2_mm"]
    gap_bot3 = beam["gap_bot3_mm"]
    incl = beam["incl_mm"]

    if 0 <= y_top < t_slab:
        return b_slab

    y_beam_top = y_top - t_slab
    y = H_beam - y_beam_top

    if H_beam - h_f_top <= y <= H_beam:
        return 2.0 * b_f_top
    if H_beam - 2.0 * h_f_top <= y < H_beam - h_f_top:
        half_width_1 = gap_top2 / 2.0
        half_width_2 = gap_top2 / 2.0 + h_f_top
        y1 = H_beam - 2.0 * h_f_top
        y2 = H_beam - h_f_top
        half_width = half_width_1 + (half_width_2 - half_width_1) * (y - y1) / (y2 - y1)- (gap_top2/2-b_w)
        return 2.0 * half_width
    if h_f_bot + incl <= y < H_beam - 2.0 * h_f_top:
        x1, y1 = b_f_bot / 2.0, 0.0
        x2, y2 = gap_top2 / 2.0, H_beam - 2.0 * h_f_top
        x3, y3 = gap_bot3 / 2.0, h_f_bot
        x4, y4 = (gap_top2-2*b_w) / 2.0, H_beam - 2.0 * h_f_top
        half_width = x1 + (x2 - x1) * (y - y1) / (y2 - y1)- (x3 + (x4 - x3) * (y - y3) / (y4 - y3))
        return 2.0 * half_width
    if h_f_bot <= y < h_f_bot + incl:
        x1, y1 = 0.0, h_f_bot
        x2, y2 = gap_bot3 / 2.0, h_f_bot + incl
        x3, y3 = b_f_bot / 2.0, 0.0
        x4, y4 = gap_top2 / 2.0, H_beam - 2.0 * h_f_top
        half_width = (x3 + (x4 - x3) * (y - y3) / (y4 - y3)) + x1 - (x2 - x1) * (y - y1) / (y2 - y1)
        return 2.0 * half_width
    if 0.0 <= y < h_f_bot:
        return b_f_bot
    return 0.0


def half_width_at_y(y: float, H: float, h_f_top: float, gap_top1: float, gap_top2: float, b_f_top: float, b_f_bot: float) -> float:
    if H - h_f_top <= y <= H:
        return gap_top1 / 2 + b_f_top
    if H - 2 * h_f_top <= y < H - h_f_top:
        y1 = H - 2 * h_f_top
        y2 = H - h_f_top
        x1 = gap_top2 / 2
        x2 = gap_top2 / 2 + h_f_top
        return x1 + (x2 - x1) * (y - y1) / (y2 - y1)
    if 0 <= y < H - 2 * h_f_top:
        y1 = 0
        y2 = H - 2 * h_f_top
        x1 = b_f_bot / 2
        x2 = gap_top2 / 2
        return x1 + (x2 - x1) * (y - y1) / (y2 - y1)
    raise ValueError(f"y = {y} mm is outside section depth")


def strand_positions(layers_df: pd.DataFrame, beam: Dict[str, float], side_cover: float) -> pd.DataFrame:
    rows = []
    for _, row in layers_df.iterrows():
        n = int(row["num_strands"])
        y = float(row["bottom_distance"])
        x_half = half_width_at_y(
            y=y,
            H=beam["H_mm"],
            h_f_top=beam["h_f_top_mm"],
            gap_top1=beam["gap_top1_mm"],
            gap_top2=beam["gap_top2_mm"],
            b_f_top=beam["b_f_top_mm"],
            b_f_bot=beam["b_f_bot_mm"],
        )
        usable_half = max(0.0, x_half - side_cover)
        x_positions = [0.0] if n <= 1 else np.linspace(-usable_half, usable_half, n)
        for x in x_positions:
            rows.append(
                {
                    "x_mm": float(x),
                    "y_mm_from_bottom": y,
                    "area_mm2": float(row["strand_area"]),
                    "layer_num_strands": n,
                }
            )
    return pd.DataFrame(rows)


# -----------------------------
# Strength calculation
# -----------------------------
def calculate_neutral_axis_bottom_ref_variable_width(
    f_c: float,
    f_pb: float,
    layers: List[Dict[str, float]],
    section_depth: float,
    width_func,
    h_f=None,
    Ep: float = 195000.0,
    eps_cu: float = 0.003,
    f_pi: float = 0.0,
    f_ps_max: float | None = None,
    tol: float = 1e-3,
    max_iter: int = 150,
    n_int: int = 400,
):
    alpha2 = max(0.67, min(1.0 - 0.003 * f_c, 0.85))
    gamma = max(0.67, min(1.05 - 0.007 * f_c, 0.85))

    if f_ps_max is None:
        f_ps_max = f_pb

    steel = []
    for layer in layers:
        A = float(layer["num_strands"] * layer["strand_area"])
        y = float(section_depth - layer["bottom_distance"])
        steel.append((A, y, float(layer["bottom_distance"])))

    def comp_resultant(x: float):
        a = gamma * x
        ys = np.linspace(0.0, a, n_int)
        bs = np.array([width_func(y) for y in ys], dtype=float)
        Aeq = np.trapezoid(bs, ys)
        Qeq = np.trapezoid(bs * ys, ys)
        C = alpha2 * f_c * Aeq
        y_c = Qeq / Aeq if Aeq > 0 else 0.0
        region = "Flange only" if (h_f is not None and a <= h_f) else "Variable width section"
        return C, y_c, a, Aeq, region

    def steel_forces(x: float):
        T_sum = 0.0
        layers_out = []
        for A, y, bottom_distance in steel:
            eps_t = eps_cu * (y - x) / max(x, 1e-9) if y > x else 0.0
            f_ps = max(f_pi, min(f_pi + Ep * eps_t, f_ps_max))
            T = A * f_ps
            T_sum += T
            layers_out.append(
                {
                    "A_mm2": A,
                    "y_mm_from_top": y,
                    "bottom_distance_mm": bottom_distance,
                    "f_ps_MPa": f_ps,
                    "T_N": T,
                    "T_kN": T / 1000.0,
                }
            )
        return T_sum, layers_out

    x = 0.2 * section_depth
    step = 0.02 * section_depth
    converged = False

    for _ in range(max_iter):
        C_c, y_c, a, Aeq, region = comp_resultant(x)
        T_sum, steel_layers = steel_forces(x)
        diff = (C_c - T_sum) / 1000.0
        if abs(diff) < tol:
            converged = True
            break
        x = x - step if diff > 0 else x + step
        x = max(1e-3, min(x, section_depth * 0.98))
        step *= 0.9

    M_u = sum(L["T_N"] * abs(L["y_mm_from_top"] - y_c) for L in steel_layers)
    total_A = sum(A for A, _, _ in steel)
    d_eff = sum(A * y for A, y, _ in steel) / total_A
    outer_bottom = min(l["bottom_distance"] for l in layers)
    d0 = section_depth - outer_bottom
    kuo = x / max(d0, 1e-9)
    raw_phi = 1.19 - (13 * kuo / 12)
    phi = max(0.6, min(raw_phi, 0.8))
    M_Rd = phi * M_u / 1e6

    return {
        "x": x,
        "a": a,
        "Aeq": Aeq,
        "y_c": y_c,
        "C_c": C_c,
        "T_sum": T_sum,
        "M_u": M_u,
        "M_Rd": M_Rd,
        "phi": phi,
        "kuo": kuo,
        "d_eff": d_eff,
        "d0": d0,
        "alpha2": alpha2,
        "gamma": gamma,
        "region": region,
        "raw_phi": raw_phi,
        "converged": converged,
        "steel_layers": steel_layers,
    }


# -----------------------------
# Plotting
# -----------------------------
def plot_section(beam: Dict[str, float], b_plate: float, t_plate: float, strands_df: pd.DataFrame):
    pts = build_super_t_points(beam)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.add_patch(Polygon(pts, closed=True, fill=False, linewidth=2))
    rect = [
        (-b_plate / 2, beam["H_mm"]),
        (b_plate / 2, beam["H_mm"]),
        (b_plate / 2, beam["H_mm"] + t_plate),
        (-b_plate / 2, beam["H_mm"] + t_plate),
    ]
    ax.add_patch(Polygon(rect, closed=True, fill=False, linewidth=2))
    if not strands_df.empty:
        ax.scatter(strands_df["x_mm"], strands_df["y_mm_from_bottom"], s=18)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y from bottom [mm]")
    ax.set_title("Composite Super-T section")
    ax.grid(True, alpha=0.3)
    return fig


def plot_width_profile(beam: Dict[str, float], b_plate: float, t_plate: float):
    H_total = beam["H_mm"] + t_plate
    ys = np.linspace(0.0, H_total, 400)
    widths = [width_composite_super_t(y, beam, b_plate, t_plate) for y in ys]
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(widths, ys)
    ax.invert_yaxis()
    ax.set_xlabel("Width [mm]")
    ax.set_ylabel("Depth from top [mm]")
    ax.set_title("Width function b(y)")
    ax.grid(True, alpha=0.3)
    return fig


# -----------------------------
# UI
# -----------------------------
st.title("Super-T Section — ULS Bending Capacity")
st.caption("by Maria Stoliarova.")
st.caption("Version with optional JSON library upload support.")

with st.sidebar.expander("Libraries (optional JSON upload)", expanded=False):

    upload_super = st.file_uploader("SuperT_library.json", type=["json"])
    upload_concrete = st.file_uploader("concrete_library.json", type=["json"])
    upload_steel = st.file_uploader("reinforcement_library.json", type=["json"])

supert_library = load_uploaded_json(upload_super) or DEFAULT_SUPERT
concrete_library = load_uploaded_json(upload_concrete) or DEFAULT_CONCRETE
steel_library = load_uploaded_json(upload_steel) or DEFAULT_STEEL

beam_options = list(strip_meta_keys(supert_library).keys())
concrete_options = list(strip_meta_keys(concrete_library).keys())
steel_options = list(strip_meta_keys(steel_library).keys())

beam_name = st.sidebar.selectbox("Beam type", beam_options)

beam_base = dict(supert_library[beam_name])   # исходные значения из библиотеки
beam = dict(beam_base)                        # рабочая копия


editor_data_key = f"layers_data_{beam_name}"
editor_widget_key = f"layers_widget_{beam_name}"

default_layers_df = pd.DataFrame(SuperT_PS_library[beam_name])

if editor_data_key not in st.session_state:
    st.session_state[editor_data_key] = default_layers_df.copy()

layers_df = pd.DataFrame(st.session_state[editor_data_key]).copy()
layers_df = layers_df.fillna(0)
layers_df = layers_df[(layers_df["num_strands"] > 0) & (layers_df["strand_area"] > 0)]




st.sidebar.header("Beam geometry [mm]")

st.header("Super-T Section Geometry")

col1, col2 = st.columns([1,1])

with col1:
    st.image("SuperT_dimentions.png", width=500)

with col2:
    b_plate = st.number_input("b_plate", value=2400.0)
    t_plate = st.number_input("t_plate", value=200.0)




# Сначала вводим все параметры, кроме зависимых
beam["b_f_top_mm"] = st.sidebar.number_input("b_f_top_mm", value=float(beam_base["b_f_top_mm"]), step=10.0)
beam["h_f_top_mm"] = st.sidebar.number_input("h_f_top_mm", value=float(beam_base["h_f_top_mm"]), step=10.0)
beam["b_f_bot_mm"] = st.sidebar.number_input("b_f_bot_mm", value=float(beam_base["b_f_bot_mm"]), step=10.0)
beam["h_f_bot_mm"] = st.sidebar.number_input("h_f_bot_mm", value=float(beam_base["h_f_bot_mm"]), step=10.0)
beam["H_mm"]       = st.sidebar.number_input("H_mm",       value=float(beam_base["H_mm"]),       step=10.0)
beam["gap_top2_mm"]= st.sidebar.number_input("gap_top2_mm",value=float(beam_base["gap_top2_mm"]),step=10.0)
beam["incl_mm"]    = st.sidebar.number_input("incl_mm",    value=float(beam_base["incl_mm"]),    step=1.0)

# b_w_mm вводим отдельно
beam["b_w_mm"] = st.sidebar.number_input("b_w_mm", value=float(beam_base["b_w_mm"]), step=10.0)

# автоматический пересчёт gap_top1_mm и gap_bot3_mm
delta_bw = beam["b_w_mm"] - beam_base["b_w_mm"]

beam["gap_top1_mm"] = beam_base["gap_top1_mm"] - 2.0 * delta_bw
beam["gap_bot3_mm"] = beam_base["gap_bot3_mm"] - 2.0 * delta_bw

# Показываем их как вычисленные значения
st.sidebar.write(f"gap_top1_mm (auto) = {beam['gap_top1_mm']:.1f}")
st.sidebar.write(f"gap_bot3_mm (auto) = {beam['gap_bot3_mm']:.1f}")




st.sidebar.header("Materials")
concrete_grade = st.sidebar.selectbox("Concrete grade", concrete_options)
steel_grade = st.sidebar.selectbox("Prestress steel", steel_options)

f_c_default = float(concrete_library[concrete_grade].get("f'c_MPa", concrete_library[concrete_grade].get("f_c_MPa", 40.0)))
f_ct_default = float(concrete_library[concrete_grade].get("fct_MPa", concrete_library[concrete_grade].get("f_ct_MPa", 0.0)))
f_pb_default = float(steel_library[steel_grade].get("f_pb_MPa", 1770.0))
E_sp_default = float(steel_library[steel_grade].get("E_MPa", steel_library[steel_grade].get("E_steel_MPa", 195000.0)))

f_c = st.sidebar.number_input("f_c [MPa]", value=f_c_default, step=1.0)
f_pb = st.sidebar.number_input("f_pb [MPa]", value=f_pb_default, step=10.0)
E_sp = st.sidebar.number_input("E_sp [MPa]", value=E_sp_default, step=1000.0)
side_cover = st.sidebar.number_input("Side cover for strand layout [mm]", value=75.0, step=5.0)

st.sidebar.header("Solver")
eps_cu = st.sidebar.number_input("eps_cu", value=0.003, step=0.0001, format="%.4f")
n_int = st.sidebar.slider("Integration strips", 100, 1200, 400, 50)

with st.expander("Library status", expanded=False):
    st.write(
        {
            "SuperT source": upload_super.name if upload_super else "embedded defaults",
            "Concrete source": upload_concrete.name if upload_concrete else "embedded defaults",
            "Steel source": upload_steel.name if upload_steel else "embedded defaults",
        }
    )
    if isinstance(concrete_library.get("meta"), dict):
        st.write({"Concrete meta": concrete_library["meta"]})
    if isinstance(steel_library.get("meta"), dict):
        st.write({"Steel meta": steel_library["meta"]})

st.subheader("Prestressing layers")
edited_layers_df = st.data_editor(
    st.session_state[editor_data_key],
    num_rows="dynamic",
    use_container_width=True,
    key=editor_widget_key,
)

st.session_state[editor_data_key] = edited_layers_df

H_total = beam["H_mm"] + t_plate
width_func = lambda y: width_composite_super_t(y, beam, b_plate, t_plate)

layers = layers_df.to_dict(orient="records")
results = calculate_neutral_axis_bottom_ref_variable_width(
    f_c=float(f_c),
    f_pb=float(f_pb),
    layers=layers,
    section_depth=H_total,
    width_func=width_func,
    Ep=float(E_sp),
    eps_cu=float(eps_cu),
    f_pi=0.0,
    f_ps_max=float(f_pb),
    h_f=t_plate,
    n_int=int(n_int),
)

beam_poly = polygon_properties(build_super_t_points(beam))
slab_rect = rectangle_properties(b_plate, t_plate, -b_plate / 2.0, beam["H_mm"])
combined = combine_properties([beam_poly, slab_rect])
strands_df = strand_positions(layers_df, beam, side_cover)

m1, m2, m3, m4 = st.columns(4)
m1.metric("M_Rd", f"{results['M_Rd']:.2f} kN·m")
m2.metric("Neutral axis x", f"{results['x']:.2f} mm")
m3.metric("φ", f"{results['phi']:.3f}")
m4.metric("ΣT", f"{results['T_sum']/1000.0:.2f} kN")

c1, c2 = st.columns([1.3, 1.0])
with c1:
    st.pyplot(plot_section(beam, b_plate, t_plate, strands_df), clear_figure=True)
with c2:
    st.pyplot(plot_width_profile(beam, b_plate, t_plate), clear_figure=True)



left, right = st.columns(2)
with left:
    st.markdown("### Geometric properties")
    st.dataframe(
        pd.DataFrame(
            {
                "Property": ["Area", "Centroid x", "Centroid y", "Ixx about centroid", "Iyy about centroid"],
                "Value": [
                    f"{combined.area:,.2f} mm²",
                    f"{combined.cx:,.2f} mm",
                    f"{combined.cy:,.2f} mm",
                    f"{combined.ixx_centroid:,.3e} mm⁴",
                    f"{combined.iyy_centroid:,.3e} mm⁴",
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.markdown("### Bending resistance summary")
    st.dataframe(
        pd.DataFrame(
            {
                "Parameter": [
                    "Section depth H_total",
                    "Compression block depth a",
                    "Equivalent compression area Aeq",
                    "Compression resultant C_c",
                    "y_c from top",
                    "d_eff",
                    "d0",
                    "kuo",
                    "alpha2",
                    "gamma",
                    "Concrete fct",
                    "M_u",
                    "M_Rd",
                ],
                "Value": [
                    f"{H_total:.2f} mm",
                    f"{results['a']:.2f} mm",
                    f"{results['Aeq']:.2f} mm²",
                    f"{results['C_c']/1000.0:.2f} kN",
                    f"{results['y_c']:.2f} mm",
                    f"{results['d_eff']:.2f} mm",
                    f"{results['d0']:.2f} mm",
                    f"{results['kuo']:.4f}",
                    f"{results['alpha2']:.3f}",
                    f"{results['gamma']:.3f}",
                    f"{f_ct_default:.2f} MPa",
                    f"{results['M_u']/1e6:.2f} kN·m",
                    f"{results['M_Rd']:.2f} kN·m",
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("### Steel layer forces")
steel_df = pd.DataFrame(results["steel_layers"])
if not steel_df.empty:
    st.dataframe(steel_df, use_container_width=True, hide_index=True)

st.markdown("## Calculation summary")

st.markdown(f"""
### Geometry
- **Section depth** H = {H_total:.2f} mm  
- **Neutral axis depth** x = {results['x']:.2f} mm from top  
- **Compression block depth** a = γ·x = {results['a']:.2f} mm  
- **Equivalent compression area** Aeq = {results['Aeq']:.2f} mm²
""")

st.markdown(f"""
### Material parameters
- **Concrete compressive strength** f'c = {f_c:.2f} MPa
""")

st.markdown(f"""
### Stress block factors
- **α₂** = {results['alpha2']:.3f}  
- **γ** = {results['gamma']:.3f}
""")

st.markdown(f"""
### Force equilibrium
- **Compression region** = {results['region']}  
- **Concrete compression resultant** C_c = {results['C_c']/1000:.2f} kN  
- **Location of compression resultant** y_c = {results['y_c']:.2f} mm from top  
- **Total prestressing force** ΣT = {results['T_sum']/1000:.2f} kN
""")

st.markdown(f"""
### Section parameters
- **Effective depth** d_eff = {results['d_eff']:.2f} mm  
- **Depth to lowest tendon** d₀ = {results['d0']:.2f} mm  
- **Neutral axis ratio** kuo = x/d₀ = {results['kuo']:.4f}
""")

st.markdown(f"""
### Strength reduction factor
- **φ_raw** = {results['raw_phi']:.3f}  
- **φ** = {results['phi']:.3f}
""")

st.markdown(f"""
### Moment resistance
- **Ultimate internal moment** M_u = {results['M_u']/1e6:.2f} kN·m  
- **Design bending resistance** M_Rd = {results['M_Rd']:.2f} kN·m
""")

st.info(
    "You can upload your JSON libraries in the sidebar. If a file is not uploaded, the app uses embedded defaults. "
    "Expected structures are the same as in your notebook: beam dictionaries in SuperT_library.json, concrete grades in concrete_library.json, and reinforcement grades in reinforcement_library.json."
)
