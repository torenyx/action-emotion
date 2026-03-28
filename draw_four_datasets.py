"""
四数据集横向柱状图（2×2 学术版）
英文标签与数值标注字号适当放大
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
from datetime import datetime
from pathlib import Path

plt.rcParams.update({
    'font.family': 'SimHei',
    'axes.unicode_minus': False,
    'font.size': 18,
})

# ---------- 数据 ----------
datasets = [
    {
        'title': '猫图像数据',
        'total': 671,
        'labels': ['Angry', 'Disgusted', 'Happy', 'Normal', 'Sad', 'Scared', 'Surprised'],
        'values': [98, 79, 98, 98, 96, 98, 100],
        'color': '#5B9BD5',
    },
    {
        'title': '狗图像数据',
        'total': 3938,
        'labels': ['Angry', 'Happy', 'Relaxed', 'Sad'],
        'values': [979, 996, 990, 973],
        'color': '#ED7D31',
    },
    {
        'title': '猫音频数据',
        'total': 2961,
        'labels': ['Angry', 'Defence', 'Fighting', 'Happy',
                   'HuntingMind', 'Mating', 'MotherCall', 'Paining', 'Resting', 'Warning'],
        'values': [300, 291, 300, 297, 289, 301, 296, 291, 296, 300],
        'color': '#4472C4',
    },
    {
        'title': '狗音频数据',
        'total': 1200,
        'labels': ['Barking', 'Growling', 'Howling', 'Whining'],
        'values': [300, 300, 300, 300],
        'color': '#C55A11',
    },
]

LABEL_FS  = 20   # y轴英文类别标签
VALUE_FS  = 20   # 柱末数值+百分比
Xtick_FS  = 20   # x轴刻度
XLABEL_FS = 20   # x轴"样本数量"
TOTAL_FS  = 20   # 右上角"共 X 样本"
TITLE_FS  = 23   # 子图标题

# ---------- 布局 ----------
fig, axes = plt.subplots(2, 2, figsize=(20, 15), facecolor='white',
                         gridspec_kw={'hspace': 0.46, 'wspace': 0.38})
axes = axes.flatten()

for ax, ds in zip(axes, datasets):
    labels = ds['labels']
    values = np.array(ds['values'])
    total  = ds['total']
    n      = len(labels)
    y      = np.arange(n)
    color  = ds['color']

    bars = ax.barh(y, values, height=0.55,
                   color=color, alpha=0.88,
                   edgecolor='white', linewidth=0.6, zorder=3)

    x_max = values.max()
    for bar, v in zip(bars, values):
        pct = v / total * 100
        ax.text(v + x_max * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f'{v}  ({pct:.1f}%)',
                va='center', ha='left', fontsize=VALUE_FS, color='#222222',
                fontfamily='Arial')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=LABEL_FS, fontfamily='Arial')
    ax.invert_yaxis()

    ax.set_xlim(0, x_max * 1.35)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6, integer=True))
    ax.tick_params(axis='x', labelsize=Xtick_FS)
    ax.set_xlabel('样本数量', fontsize=XLABEL_FS)

    ax.set_title(ds['title'], fontsize=TITLE_FS, fontweight='bold',
                 color='#1a1a1a', loc='left', pad=10)
    ax.text(1.0, 1.025,
            f'共 {total:,} 样本',
            transform=ax.transAxes,
            ha='right', va='bottom', fontsize=TOTAL_FS, color='#555555')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)
    ax.set_facecolor('white')

# ---------- 图例 ----------
legend_items = [
    mpatches.Patch(color='#5B9BD5', alpha=0.88, label='猫 — 图像'),
    mpatches.Patch(color='#4472C4', alpha=0.88, label='猫 — 音频'),
    mpatches.Patch(color='#ED7D31', alpha=0.88, label='狗 — 图像'),
    mpatches.Patch(color='#C55A11', alpha=0.88, label='狗 — 音频'),
]
fig.legend(handles=legend_items,
           loc='lower center', ncol=4,
           fontsize=20, frameon=False,
           bbox_to_anchor=(0.5, -0.01))

# ---------- 保存 ----------
ts  = datetime.now().strftime('%Y%m%d%H%M%S')
out = Path('d:/bcode/pet_result/figure') / f'four_datasets_detail_{ts}.png'
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {out}')
