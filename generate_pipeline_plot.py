#!/usr/bin/env python3
"""
Generate SynCrash Pipeline Architecture diagram for CVPR workshop poster.
Produces both PNG (300 dpi) and PDF (vector).
"""
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

# ── colour palette (dark-blue academic style) ──────────────────────────────
C_BG       = '#FFFFFF'
C_INPUT    = '#D6EAF8'   # light blue  – input node
C_TEMPORAL = '#2E86C1'   # medium blue – temporal stage
C_SPATIAL  = '#1B4F72'   # dark blue   – spatial stage
C_OUTPUT   = '#117A65'   # teal-green  – final output
C_ARROW    = '#2C3E50'
C_LABEL    = '#2C3E50'
C_WHITE    = '#FFFFFF'
C_DARK     = '#1C2833'

# ── layout constants ───────────────────────────────────────────────────────
FIG_W, FIG_H = 10, 18          # figure size in inches
BOX_W, BOX_H = 5.0, 0.95       # node width / height
X_CENTER     = 5.0              # horizontal center of the flow
Y_TOP        = 16.0             # y of first node centre
Y_STEP       = 1.65             # vertical gap between node centres
CORNER_RAD   = 0.25             # rounded-corner radius

# ── node definitions (label, bg colour, text colour, optional subtitle) ────
nodes = [
    ("CCTV Video Input",                          C_INPUT,    C_DARK,  None),
    ("VideoMAEv2-Giant\n(1408-d backbone)",        C_TEMPORAL, C_WHITE, "16-frame clips, stride 8"),
    ("+ Scene / Weather / Daytime\nEmbeddings (96-d)", C_TEMPORAL, C_WHITE, "→ 1504-d combined feature"),
    ("Binary Logit Head\n(BCEWithLogitsLoss)",     C_TEMPORAL, C_WHITE, "AdamW, lr = 2e-5"),
    ("Dense Sliding Window\nInference (stride = 2)", C_TEMPORAL, C_WHITE, "Batched FP16"),
    ("Gaussian Smoothing (σ = 2)\n→ argmax → Accident Time", C_TEMPORAL, C_WHITE, None),
    ("YOLO (conf = 0.15) + ByteTrack\n→ JSON 2D BBox Cache", C_SPATIAL, C_WHITE, None),
    ("6-Strategy Spatial Cascade\n(Overlap → Ray → SWM → Closest → Single → 0.5)", C_SPATIAL, C_WHITE, "5-frame Polyfit velocity"),
    ("Impact Point\n(center_x, center_y)",         C_SPATIAL,  C_WHITE, None),
    ("Rule-Based Type Classification\n(Impact-Angle Kinematic Heuristic)", C_OUTPUT, C_WHITE, "single | rear-end | t-bone | sideswipe | head-on"),
]


def draw():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=C_BG)
    ax.set_facecolor(C_BG)

    # explicitly set axis limits so everything is visible
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect('equal')
    ax.axis('off')

    y_positions = []

    # ── draw nodes ─────────────────────────────────────────────────────────
    for i, (label, bg, tc, subtitle) in enumerate(nodes):
        cx = X_CENTER
        cy = Y_TOP - i * Y_STEP
        y_positions.append(cy)

        # rounded rectangle
        rect = mpatches.FancyBboxPatch(
            (cx - BOX_W / 2, cy - BOX_H / 2),
            BOX_W, BOX_H,
            boxstyle=f"round,pad=0.05,rounding_size={CORNER_RAD}",
            facecolor=bg, edgecolor=C_ARROW, linewidth=1.5,
            zorder=3,
        )
        ax.add_patch(rect)

        # main label
        ax.text(cx, cy + (0.08 if subtitle else 0), label,
                ha='center', va='center', fontsize=10, fontweight='bold',
                color=tc, zorder=4, linespacing=1.3)

        # subtitle (smaller, italic)
        if subtitle:
            ax.text(cx, cy - 0.30, subtitle,
                    ha='center', va='center', fontsize=8, fontstyle='italic',
                    color=tc, alpha=0.85, zorder=4)

    # ── draw arrows between consecutive nodes ──────────────────────────────
    for i in range(len(y_positions) - 1):
        y_from = y_positions[i] - BOX_H / 2 - 0.04
        y_to   = y_positions[i + 1] + BOX_H / 2 + 0.04
        ax.annotate(
            '', xy=(X_CENTER, y_to), xytext=(X_CENTER, y_from),
            arrowprops=dict(
                arrowstyle='-|>', color=C_ARROW, lw=2.0,
                connectionstyle='arc3,rad=0',
            ),
            zorder=2,
        )

    # ── stage brackets on the left side ────────────────────────────────────
    def add_bracket(y_top, y_bot, label, colour):
        """Draw a vertical bracket with a rotated label."""
        bx = X_CENTER - BOX_W / 2 - 0.7
        ax.annotate('', xy=(bx, y_bot - BOX_H / 2 + 0.1),
                    xytext=(bx, y_top + BOX_H / 2 - 0.1),
                    arrowprops=dict(arrowstyle='-', color=colour, lw=2.5))
        # top tick
        ax.plot([bx, bx + 0.15], [y_top + BOX_H / 2 - 0.1]*2, color=colour, lw=2.5)
        # bottom tick
        ax.plot([bx, bx + 0.15], [y_bot - BOX_H / 2 + 0.1]*2, color=colour, lw=2.5)
        # label
        mid_y = (y_top + y_bot) / 2
        ax.text(bx - 0.15, mid_y, label,
                ha='center', va='center', fontsize=11, fontweight='bold',
                color=colour, rotation=90, zorder=5)

    add_bracket(y_positions[1], y_positions[5], "Stage 1-2\nTemporal", C_TEMPORAL)
    add_bracket(y_positions[6], y_positions[8], "Stage 3\nSpatial",   C_SPATIAL)

    # ── title ──────────────────────────────────────────────────────────────
    ax.text(X_CENTER, Y_TOP + 1.0,
            "SynCrash: Modular Pipeline Architecture",
            ha='center', va='center', fontsize=16, fontweight='bold',
            color=C_DARK, zorder=5)

    ax.text(X_CENTER, Y_TOP + 0.6,
            "Accident Detection, Localization & Classification",
            ha='center', va='center', fontsize=11, fontstyle='italic',
            color='#5D6D7E', zorder=5)

    # ── save ───────────────────────────────────────────────────────────────
    out_png = "/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4/syncrash_pipeline.png"
    out_pdf = "/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4/syncrash_pipeline.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor=C_BG)
    fig.savefig(out_pdf, bbox_inches='tight', facecolor=C_BG)
    plt.close(fig)
    print(f"✅ Saved: {out_png}")
    print(f"✅ Saved: {out_pdf}")


if __name__ == "__main__":
    draw()
