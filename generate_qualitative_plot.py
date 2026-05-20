import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_qualitative_plot():
    # Larger figure for HD quality, more breathing room
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), facecolor='white')
    fig.subplots_adjust(wspace=0.15)

    # =================================================================
    # Panel 1: SUCCESS CASE — T-Bone at Intersection
    # =================================================================
    ax1 = axes[0]
    ax1.set_facecolor('#EAECEE')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title("Success Case: Clear T-Bone Localization",
                  fontsize=18, fontweight='bold', color='#1E8449', pad=20)

    # Road / Intersection background
    ax1.fill_between([3.2, 6.8], 0, 10, color='#D5D8DC', zorder=0)
    ax1.fill_between([0, 10], 3.2, 6.8, color='#D5D8DC', zorder=0)
    # Lane markings
    for seg in [(5, 0, 5, 3.2), (5, 6.8, 5, 10)]:
        ax1.plot([seg[0], seg[2]], [seg[1], seg[3]],
                 color='white', linestyle='--', linewidth=2.5, zorder=1)
    for seg in [(0, 5, 3.2, 5), (6.8, 5, 10, 5)]:
        ax1.plot([seg[0], seg[2]], [seg[1], seg[3]],
                 color='white', linestyle='--', linewidth=2.5, zorder=1)

    # ── Car A (Eastbound) ──
    car_a = patches.FancyBboxPatch(
        (1.2, 4.2), 2.0, 0.9,
        boxstyle="round,pad=0.08", linewidth=2.5,
        edgecolor='#2874A6', facecolor='#AED6F1', zorder=3)
    ax1.add_patch(car_a)
    ax1.text(2.2, 4.65, "Car A", color='#154360', fontsize=12,
             fontweight='bold', ha='center', va='center', zorder=4)
    # Velocity arrow
    ax1.annotate('', xy=(4.6, 4.65), xytext=(3.3, 4.65),
                 arrowprops=dict(facecolor='#2874A6', edgecolor='#2874A6',
                                 width=3, headwidth=12, headlength=8),
                 zorder=4)
    ax1.text(3.9, 4.1, r'$v_A$', fontsize=14, fontweight='bold',
             color='#2874A6', ha='center', zorder=4)

    # ── Car B (Northbound) ──
    car_b = patches.FancyBboxPatch(
        (5.1, 1.2), 0.9, 2.0,
        boxstyle="round,pad=0.08", linewidth=2.5,
        edgecolor='#B03A2E', facecolor='#F5B7B1', zorder=3)
    ax1.add_patch(car_b)
    ax1.text(5.55, 2.2, "Car B", color='#641E16', fontsize=12,
             fontweight='bold', ha='center', va='center', rotation=90, zorder=4)
    # Velocity arrow
    ax1.annotate('', xy=(5.55, 4.5), xytext=(5.55, 3.3),
                 arrowprops=dict(facecolor='#B03A2E', edgecolor='#B03A2E',
                                 width=3, headwidth=12, headlength=8),
                 zorder=4)
    ax1.text(6.1, 3.8, r'$v_B$', fontsize=14, fontweight='bold',
             color='#B03A2E', ha='center', zorder=4)

    # ── Impact Point ──
    ax1.plot(5.0, 4.9, marker='X', color='#F39C12', markersize=22,
             markeredgecolor='black', markeredgewidth=2, zorder=5)
    ax1.text(5.0, 5.7, "Predicted\nImpact Point", color='#B7950B',
             fontsize=13, fontweight='bold', ha='center', va='bottom',
             bbox=dict(facecolor='#FEF9E7', edgecolor='#F39C12',
                       boxstyle='round,pad=0.4', alpha=0.95),
             zorder=5)

    # ── Info box ──
    info1 = ("Kinematic Features:\n"
             "  - Impact angle  \u03B8 \u2248 90\u00B0\n"
             "  - Strategy: B (Ray Intersection)\n"
             "\n"
             "Prediction:  t-bone")
    ax1.text(0.3, 9.7, info1, fontsize=12, fontweight='bold', color='#1C2833',
             va='top', linespacing=1.5,
             bbox=dict(facecolor='white', alpha=0.92, edgecolor='#BDC3C7',
                       boxstyle='round,pad=0.6'), zorder=5)

    # =================================================================
    # Panel 2: FAILURE CASE — Night / Occlusion
    # =================================================================
    ax2 = axes[1]
    ax2.set_facecolor('#1B2631')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title("Failure Case: Night / Severe Occlusion",
                  fontsize=18, fontweight='bold', color='#E74C3C', pad=20)

    # Road (dark)
    ax2.fill_between([3.2, 6.8], 0, 10, color='#212F3D', zorder=0)
    ax2.plot([5, 5], [0, 10], color='#5D6D7E', linestyle='--',
             linewidth=2, zorder=1)

    # Dim headlight glow effect
    glow = patches.Circle((4.5, 5.0), 2.5, color='#F7DC6F', alpha=0.06, zorder=1)
    ax2.add_patch(glow)
    glow2 = patches.Circle((4.5, 5.0), 1.5, color='#F7DC6F', alpha=0.08, zorder=1)
    ax2.add_patch(glow2)

    # ── Car A (Northbound, Detected) ──
    car_c = patches.FancyBboxPatch(
        (3.9, 2.0), 1.0, 2.2,
        boxstyle="round,pad=0.08", linewidth=2.5,
        edgecolor='#2E86C1', facecolor='#85C1E9', zorder=3)
    ax2.add_patch(car_c)
    ax2.text(4.4, 3.1, "Detected", color='#1B4F72', fontsize=11,
             fontweight='bold', ha='center', va='center', rotation=90, zorder=4)
    # Velocity arrow
    ax2.annotate('', xy=(4.4, 5.2), xytext=(4.4, 4.3),
                 arrowprops=dict(facecolor='#2E86C1', edgecolor='#2E86C1',
                                 width=3, headwidth=12, headlength=8),
                 zorder=4)
    ax2.text(3.5, 4.7, r'$v_A$', fontsize=14, fontweight='bold',
             color='#5DADE2', ha='center', zorder=4)

    # ── Car B (Southbound, MISSED — dashed outline) ──
    car_d = patches.FancyBboxPatch(
        (5.1, 7.0), 1.0, 2.2,
        boxstyle="round,pad=0.08", linewidth=2.5,
        edgecolor='#95A5A6', facecolor='none', linestyle=':', zorder=3)
    ax2.add_patch(car_d)
    ax2.text(5.6, 8.1, "Missed\n(Dark)", color='#ABB2B9', fontsize=11,
             fontweight='bold', ha='center', va='center', rotation=90, zorder=4)
    # No velocity arrow (undetected)

    # ── Fallback Impact Point ──
    ax2.plot(5.0, 5.0, marker='X', color='#E74C3C', markersize=22,
             markeredgecolor='white', markeredgewidth=2, zorder=5)
    ax2.text(6.5, 5.0, "Fallback Center\n(0.5, 0.5)", color='white',
             fontsize=14, fontweight='bold', ha='left', va='center',
             bbox=dict(facecolor='#E74C3C', edgecolor='white',
                       boxstyle='round,pad=0.5', alpha=0.9),
             zorder=5)
    # Arrow from X to label
    ax2.annotate('', xy=(6.4, 5.0), xytext=(5.4, 5.0),
                 arrowprops=dict(arrowstyle='->', color='white', lw=2),
                 zorder=5)

    # ── Info box ──
    info2 = ("Kinematic Features:\n"
             "  - Detected vehicles: 1\n"
             "  - 2nd vehicle occluded\n"
             "  - Strategy: F (Fallback)\n"
             "\n"
             "Prediction:  single-vehicle")
    ax2.text(0.3, 9.7, info2, fontsize=12, fontweight='bold', color='#F2F3F4',
             va='top', linespacing=1.5,
             bbox=dict(facecolor='#17202A', alpha=0.9, edgecolor='#7F8C8D',
                       boxstyle='round,pad=0.6'), zorder=5)

    # ── Save ──
    out_png = "/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4/qualitative_localization.png"
    out_pdf = "/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4/qualitative_localization.pdf"
    fig.savefig(out_png, dpi=400, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    draw_qualitative_plot()
