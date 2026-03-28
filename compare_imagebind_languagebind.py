# -*- coding: utf-8 -*-
"""
ImageBind vs LanguageBind 多任务分类结果对比可视化
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime
import os

# ── 中文字体设置 ──────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ── 数据定义 ──────────────────────────────────────────────────
# ImageBind 测试集结果
IB = {
    'dog_img': {
        'acc': 87.06,
        'macro_p': 87.41, 'macro_r': 87.07, 'macro_f1': 87.11,
        'classes': ['angry', 'happy', 'relaxed', 'sad'],
        'f1': [90.91, 86.01, 83.81, 87.70],
        'precision': [90.00, 89.25, 79.28, 91.11],
        'recall': [91.84, 83.00, 88.89, 84.54],
    },
    'cat_img': {
        'acc': 85.07,
        'macro_p': 86.92, 'macro_r': 85.36, 'macro_f1': 85.18,
        'classes': ['Angry', 'Disgusted', 'Happy', 'Normal', 'Sad', 'Scared', 'Surprised'],
        'f1': [88.89, 82.35, 94.74, 84.21, 78.26, 77.78, 90.00],
        'precision': [100.00, 77.78, 100.00, 88.89, 64.29, 87.50, 90.00],
        'recall': [80.00, 87.50, 90.00, 80.00, 100.00, 70.00, 90.00],
    },
    'dog_audio': {
        'acc': 98.33,
        'macro_p': 98.39, 'macro_r': 98.33, 'macro_f1': 98.32,
        'classes': ['barking', 'growling', 'howling', 'whining'],
        'f1': [96.55, 98.36, 98.36, 100.00],
        'precision': [100.00, 96.77, 96.77, 100.00],
        'recall': [93.33, 100.00, 100.00, 100.00],
    },
    'cat_audio': {
        'acc': 85.81,
        'macro_p': 86.54, 'macro_r': 85.87, 'macro_f1': 85.93,
        'classes': ['Angry', 'Defence', 'Fighting', 'Happy', 'HuntingMind',
                    'Mating', 'MotherCall', 'Paining', 'Resting', 'Warning'],
        'f1': [85.19, 96.67, 90.00, 74.63, 94.74, 89.29, 81.97, 73.68, 91.80, 81.36],
        'precision': [95.83, 93.55, 90.00, 67.57, 96.43, 96.15, 80.65, 75.00, 87.50, 82.76],
        'recall': [76.67, 100.00, 90.00, 83.33, 93.10, 83.33, 83.33, 72.41, 96.55, 80.00],
    },
}

# LanguageBind 测试集结果
LB = {
    'dog_img': {
        'acc': 88.58,
        'macro_p': 88.89, 'macro_r': 88.53, 'macro_f1': 88.55,
        'classes': ['angry', 'happy', 'relaxed', 'sad'],
        'f1': [90.16, 89.32, 87.92, 86.81],
        'precision': [91.58, 86.79, 84.26, 92.94],
        'recall': [88.78, 92.00, 91.92, 81.44],
    },
    'cat_img': {
        'acc': 83.58,
        'macro_p': 84.51, 'macro_r': 83.57, 'macro_f1': 83.54,
        'classes': ['Angry', 'Disgusted', 'Happy', 'Normal', 'Sad', 'Scared', 'Surprised'],
        'f1': [94.74, 70.59, 94.74, 72.73, 90.00, 84.21, 77.78],
        'precision': [100.00, 66.67, 100.00, 66.67, 81.82, 88.89, 87.50],
        'recall': [90.00, 75.00, 90.00, 80.00, 100.00, 80.00, 70.00],
    },
    'dog_audio': {
        'acc': 100.00,
        'macro_p': 100.00, 'macro_r': 100.00, 'macro_f1': 100.00,
        'classes': ['barking', 'growling', 'howling', 'whining'],
        'f1': [100.00, 100.00, 100.00, 100.00],
        'precision': [100.00, 100.00, 100.00, 100.00],
        'recall': [100.00, 100.00, 100.00, 100.00],
    },
    'cat_audio': {
        'acc': 91.22,
        'macro_p': 91.42, 'macro_r': 91.25, 'macro_f1': 91.27,
        'classes': ['Angry', 'Defence', 'Fighting', 'Happy', 'HuntingMind',
                    'Mating', 'MotherCall', 'Paining', 'Resting', 'Warning'],
        'f1': [88.14, 98.31, 98.36, 83.87, 98.25, 87.72, 93.33, 82.14, 100.00, 82.54],
        'precision': [89.66, 96.67, 96.77, 81.25, 100.00, 92.59, 93.33, 85.19, 100.00, 78.79],
        'recall': [86.67, 100.00, 100.00, 86.67, 96.55, 83.33, 93.33, 79.31, 100.00, 86.67],
    },
}

TASK_NAMES = {
    'dog_img': '狗图像情感',
    'cat_img': '猫图像情感',
    'dog_audio': '狗音频分类',
    'cat_audio': '猫音频分类',
}
TASKS = list(TASK_NAMES.keys())

# 配色
C_IB = '#2878B5'    # ImageBind 蓝
C_LB = '#E84545'    # LanguageBind 红

# ── 时间戳 ────────────────────────────────────────────────────
ts = datetime.now().strftime('%Y%m%d%H%M%S')
os.makedirs('figure', exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 图1：四任务总览 — 准确率、宏平均 F1（不含加权 F1）
# ══════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 7), facecolor='white')
metrics = ['acc', 'macro_f1']
metric_labels = ['准确率 (%)', '宏平均 F1 (%)']

x = np.arange(len(TASKS))
w = 0.32

for ax, met, mlabel in zip(axes1, metrics, metric_labels):
    ib_vals = [IB[t][met] for t in TASKS]
    lb_vals = [LB[t][met] for t in TASKS]

    bars_ib = ax.bar(x - w/2, ib_vals, width=w, color=C_IB, alpha=0.88, label='ImageBind',
                     zorder=3, edgecolor='white', linewidth=0.8)
    bars_lb = ax.bar(x + w/2, lb_vals, width=w, color=C_LB, alpha=0.88, label='LanguageBind',
                     zorder=3, edgecolor='white', linewidth=0.8)

    # 数值标注
    for bar in bars_ib:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f'{bar.get_height():.1f}', ha='center', va='bottom',
                fontsize=13, color=C_IB, fontweight='bold')
    for bar in bars_lb:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f'{bar.get_height():.1f}', ha='center', va='bottom',
                fontsize=13, color=C_LB, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_NAMES[t] for t in TASKS], fontsize=16)
    ax.set_ylabel(mlabel, fontsize=17)
    ax.set_ylim(70, 106)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('white')

    # 差值标注箭头
    for i, (iv, lv) in enumerate(zip(ib_vals, lb_vals)):
        diff = lv - iv
        color = C_LB if diff > 0 else C_IB
        sign = '+' if diff > 0 else ''
        ax.annotate(f'{sign}{diff:.1f}', xy=(i, max(iv, lv) + 1.8),
                    ha='center', va='bottom', fontsize=12,
                    color=color, fontweight='bold',
                    arrowprops=None)

    ax.legend(fontsize=15, loc='lower right', framealpha=0.9)

fig1.tight_layout(pad=2.5)
out1 = f'figure/compare_overview_{ts}.png'
fig1.savefig(out1, dpi=180, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print(f'[✓] 总览对比图已保存：{out1}')

# ══════════════════════════════════════════════════════════════
# 图2：综合汇总热力表（不含加权 F1）
# ══════════════════════════════════════════════════════════════
rows = ['准确率', '宏精确率', '宏召回率', '宏F1']
row_keys = ['acc', 'macro_p', 'macro_r', 'macro_f1']

fig4, ax4 = plt.subplots(figsize=(16, 5.2), facecolor='white')
ax4.set_facecolor('white')
ax4.axis('off')

col_headers = []
for t in TASKS:
    col_headers.append(f'{TASK_NAMES[t]}\nImageBind')
    col_headers.append(f'{TASK_NAMES[t]}\nLanguageBind')

cell_text = []
cell_colors = []
for rk in row_keys:
    row_vals = []
    row_cols = []
    for t in TASKS:
        iv = IB[t][rk]
        lv = LB[t][rk]
        row_vals.append(f'{iv:.2f}%')
        row_vals.append(f'{lv:.2f}%')
        # 比较上色
        if lv > iv:
            row_cols.append('#D6EAF8')   # IB 略淡蓝
            row_cols.append('#FADBD8')   # LB 更优 红
        elif lv < iv:
            row_cols.append('#D5F5E3')   # IB 更优 绿
            row_cols.append('#FDFEFE')
        else:
            row_cols.append('#FDFEFE')
            row_cols.append('#FDFEFE')
    cell_text.append(row_vals)
    cell_colors.append(row_cols)

table = ax4.table(
    cellText=cell_text,
    rowLabels=rows,
    colLabels=col_headers,
    cellColours=cell_colors,
    cellLoc='center',
    loc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1.0, 2.6)

# 加粗列标题
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_text_props(fontweight='bold', fontsize=12)
    if c == -1:
        cell.set_text_props(fontweight='bold', fontsize=13)

# 图例说明
legend_patches = [
    mpatches.Patch(color='#FADBD8', label='LanguageBind 更优'),
    mpatches.Patch(color='#D5F5E3', label='ImageBind 更优'),
]
ax4.legend(handles=legend_patches, loc='upper right',
           bbox_to_anchor=(1.0, 1.05), fontsize=13, framealpha=0.9)

fig4.tight_layout(pad=0.5)
out4 = f'figure/compare_table_{ts}.png'
fig4.savefig(out4, dpi=180, bbox_inches='tight', facecolor='white')
plt.close(fig4)
print(f'[✓] 汇总对比表已保存：{out4}')

print('\n总览图与汇总表（无加权 F1）绘制完成。')
