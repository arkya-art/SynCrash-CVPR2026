import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def draw_evolution_plot():
    # Data
    methods = [
        'ViViT Joint Model', 
        'RAFT-based Flow', 
        'VideoMAEv2 + GradCAM', 
        'Q-Former Localization', 
        'YOLO + Physics Heuristic\n(Ours Final)'
    ]
    scores = [0.28, 0.29, 0.34, 0.37, 0.40]
    
    # Insights to annotate (index -> (text, colour))
    insights = {
        0: ("Poor spatial stability", "#E74C3C"),      # Red
        2: ("Better temporal reasoning", "#F39C12"),   # Orange
        4: ("Best overall robustness", "#27AE60")      # Green
    }

    # Setup figure — extra width so annotation boxes don't crowd the bars
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
    ax.set_facecolor('#F8F9F9')

    # Bar colours
    colors = ['#BDC3C7', '#BDC3C7', '#5DADE2', '#5DADE2', '#28B463']

    # Horizontal bars
    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, scores, height=0.6, color=colors, edgecolor='none')

    # Axes styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=11, fontweight='bold', color='#2C3E50')
    ax.set_xlabel('Overall Evaluation Score', fontsize=12, fontweight='bold',
                  color='#2C3E50', labelpad=10)
    ax.set_xlim(0.20, 0.58)   # wider x-range so annotations sit comfortably

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#BDC3C7')

    ax.xaxis.grid(True, linestyle='--', alpha=0.6, color='#BDC3C7')
    ax.set_axisbelow(True)

    # Score labels + insight annotation boxes
    for i, bar in enumerate(bars):
        width = bar.get_width()
        bar_cy = bar.get_y() + bar.get_height() / 2

        # Score number right next to the bar end
        ax.text(width + 0.005, bar_cy, f'{width:.2f}',
                ha='left', va='center', fontsize=12, fontweight='bold',
                color='#2C3E50')

        # Insight boxes — placed well to the right so they don't touch scores
        if i in insights:
            text, color = insights[i]
            # x position of the annotation box (far right of the score number)
            ann_x = width + 0.08          # gap between score number and box
            ax.annotate(
                text,
                xy=(width + 0.03, bar_cy),          # arrow starts near score
                xytext=(ann_x, bar_cy),              # box sits further right
                arrowprops=dict(arrowstyle="-", color=color, lw=1.5),
                ha='left', va='center', fontsize=11, fontweight='bold',
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=color, lw=1.5),
            )

    # ── Title & subtitle with clear vertical separation ───────────────────
    ax.set_title("Design Evolution & Method Progression",
                 fontsize=16, fontweight='bold', color='#1C2833',
                 pad=30)                              # big pad pushes title up

    fig.text(0.5, 0.92,
             "Progressive improvement from baseline to modular heuristic approach",
             ha='center', fontsize=11, fontstyle='italic', color='#5D6D7E')

    plt.tight_layout(rect=[0, 0, 1, 0.90])   # leave top 10 % for title area

    # Save
    out_png = "/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4/design_evolution.png"
    out_pdf = "/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4/design_evolution.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out_png}")
    print(f"✅ Saved: {out_pdf}")

if __name__ == "__main__":
    draw_evolution_plot()
