"""
DSMF (Decision-level Semantic Mapping Fusion) 框架流程图 - 科研级
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(18, 26))
ax.set_xlim(0, 18)
ax.set_ylim(0, 26)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

# ============================================================
# 颜色方案
# ============================================================
C_IMG       = '#1565C0';  C_IMG_BG    = '#E3F2FD'
C_AUD       = '#D84315';  C_AUD_BG    = '#FBE9E7'
C_SPECIES   = '#6A1B9A';  C_SPECIES_BG= '#F3E5F5'
C_MAP       = '#2E7D32';  C_MAP_BG    = '#E8F5E9'
C_UESS      = '#283593';  C_UESS_BG   = '#E8EAF6'
C_CONF      = '#AD1457';  C_CONF_BG   = '#FCE4EC'
C_WEIGHT    = '#E65100';  C_WEIGHT_BG = '#FFF3E0'
C_FUSION    = '#00695C';  C_FUSION_BG = '#E0F7FA'
C_PAIN      = '#B71C1C';  C_PAIN_BG   = '#FFCDD2'
C_OUT       = '#1B5E20';  C_OUT_BG    = '#C8E6C9'
C_CONSIST   = '#37474F';  C_CONSIST_BG= '#ECEFF1'
C_LABEL     = '#37474F'
C_ARROW     = '#546E7A'

# ============================================================
# 辅助函数
# ============================================================
def draw_box(ax, cx, cy, w, h, text, fc, ec, fontcolor,
             fontsize=14, fontweight='normal', rounded=True,
             linewidth=2.0):
    style = f'round,pad=0.12,rounding_size=0.15' if rounded else 'square,pad=0.05'
    box = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=style,
        facecolor=fc, edgecolor=ec, linewidth=linewidth,
        zorder=3
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fontsize, color=fontcolor, fontweight=fontweight,
            zorder=4, linespacing=1.45)

def draw_diamond(ax, cx, cy, w, h, text, fc, ec, fontcolor, fontsize=13):
    diamond_x = [cx, cx + w/2, cx, cx - w/2, cx]
    diamond_y = [cy + h/2, cy, cy - h/2, cy, cy + h/2]
    ax.fill(diamond_x, diamond_y, fc=fc, ec=ec, linewidth=2.0, zorder=3)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fontsize, color=fontcolor, fontweight='bold',
            zorder=4, linespacing=1.4)

def arrow_straight(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.8,
                   dashed=False, head_w=10, head_l=8):
    ls = '-' if not dashed else (0, (6, 4))
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=f'-|>,head_width=0.25,head_length=0.18',
                    lw=lw, color=color, linestyle=ls,
                    shrinkA=2, shrinkB=2,
                ),
                zorder=2)

def arrow_path(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.8,
               dashed=False, connectionstyle='arc3,rad=0'):
    ls = '-' if not dashed else (0, (6, 4))
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>,head_width=0.25,head_length=0.18',
        mutation_scale=12,
        lw=lw, color=color,
        linestyle=ls,
        connectionstyle=connectionstyle,
        shrinkA=2, shrinkB=2,
        zorder=2
    )
    ax.add_patch(a)

def stage_label(ax, y, text, x=1.2):
    ax.text(x, y, text, ha='center', va='center',
            fontsize=20, color=C_LABEL, fontstyle='italic',
            fontweight='bold', zorder=4,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#BDBDBD',
                      lw=0.8, alpha=0.9))

# ============================================================
# 布局
# ============================================================
LX = 5.0       # 左列中心
RX = 13.0      # 右列中心
CX = 9.0       # 中心
BW = 4.0       # 盒子宽
BH = 1.0       # 盒子高

Y = {
    'input':   24.5,
    'model':   23.0,
    'softmax': 21.5,
    'map':     20.0,
    'uess':    18.5,
    'conf':    17.0,
    'weight':  15.5,
    'fusion':  14.0,
    'pain':    12.2,
    'output':  10.4,
    'risk':    9.0,
}

# ============================================================
# 绘制节点
# ============================================================

# --- 输入层 ---
draw_box(ax, LX, Y['input'], 3.5, 0.85, '宠物图像输入',
         C_IMG_BG, C_IMG, C_IMG, 20, 'bold')
draw_box(ax, RX, Y['input'], 3.5, 0.85, '宠物音频输入',
         C_AUD_BG, C_AUD, C_AUD, 20, 'bold')

# --- 模型推理 ---
draw_box(ax, LX, Y['model'], BW, BH, '图像情绪识别模型',
         C_IMG_BG, C_IMG, C_IMG, 19)
draw_box(ax, RX, Y['model'], BW, BH, '音频行为识别模型',
         C_AUD_BG, C_AUD, C_AUD, 19)

# --- 物种路由 (菱形) ---
draw_diamond(ax, CX, Y['model'], 3.4, 1.5,
             '物种二阶段路由\n与强制对齐',
             C_SPECIES_BG, C_SPECIES, C_SPECIES, 16)

# --- Softmax ---
draw_box(ax, LX, Y['softmax'], BW, BH,
         'Softmax 概率分布\n(情绪标签)',
         C_IMG_BG, C_IMG, C_IMG, 18)
draw_box(ax, RX, Y['softmax'], BW, BH,
         'Softmax 概率分布\n(行为标签)',
         C_AUD_BG, C_AUD, C_AUD, 18)

# --- 软映射矩阵 ---
draw_box(ax, LX, Y['map'], BW, BH,
         '行为学先验\n软映射矩阵',
         C_MAP_BG, C_MAP, C_MAP, 18, 'bold')
draw_box(ax, RX, Y['map'], BW, BH,
         '行为学先验\n软映射矩阵',
         C_MAP_BG, C_MAP, C_MAP, 18, 'bold')

# --- UESS 投射 ---
draw_box(ax, LX, Y['uess'], BW, BH,
         'UESS 投射分布',
         C_UESS_BG, C_UESS, C_UESS, 19)
draw_box(ax, RX, Y['uess'], BW, BH,
         'UESS 投射分布',
         C_UESS_BG, C_UESS, C_UESS, 19)

# UESS 中心标注
ax.text(CX, Y['uess'] + 0.7, '统一情绪状态空间 (UESS)',
        ha='center', va='center', fontsize=17, color=C_UESS,
        fontweight='bold', zorder=4)

# --- 置信度 ---
draw_box(ax, LX, Y['conf'], BW, BH,
         '归一化信息熵\n确定性得分',
         C_CONF_BG, C_CONF, C_CONF, 18)
draw_box(ax, RX, Y['conf'], BW, BH,
         '归一化信息熵\n确定性得分',
         C_CONF_BG, C_CONF, C_CONF, 18)

# --- 自适应权重 ---
draw_box(ax, CX, Y['weight'], 6.5, 1.05,
         '置信度自适应加权\n(含 barking 歧义惩罚 & 下界裁剪)',
         C_WEIGHT_BG, C_WEIGHT, C_WEIGHT, 18, 'bold')

# --- 加权融合 ---
draw_box(ax, CX, Y['fusion'], 5.8, 1.05,
         '加权线性融合',
         C_FUSION_BG, C_FUSION, C_FUSION, 20, 'bold')

# --- 模态一致性 ---
draw_box(ax, 15.8, Y['fusion'], 3.0, 0.95,
         '模态一致性评估\n(余弦相似度)',
         C_CONSIST_BG, C_CONSIST, C_CONSIST, 15, linewidth=1.5)

# --- Paining 覆盖 (菱形) ---
draw_diamond(ax, CX, Y['pain'], 4.0, 1.7,
             '猫类疼痛覆盖',
             C_PAIN_BG, C_PAIN, C_PAIN, 18)

# --- 最终输出 ---
draw_box(ax, CX, Y['output'], 5.8, 0.95,
         '主情绪状态 + 特殊行为标注',
         C_OUT_BG, C_OUT, C_OUT, 20, 'bold')

# --- 风险等级 ---
draw_box(ax, CX, Y['risk'], 4.0, 0.9,
         '风险等级评估',
         C_OUT_BG, C_OUT, C_OUT, 20, 'bold')

# ============================================================
# 绘制箭头
# ============================================================
hBH = BH / 2
hBH2 = 0.85 / 2

# 输入 -> 模型
arrow_straight(ax, LX, Y['input'] - hBH2, LX, Y['model'] + hBH, C_IMG)
arrow_straight(ax, RX, Y['input'] - hBH2, RX, Y['model'] + hBH, C_AUD)

# 模型 -> Softmax
arrow_straight(ax, LX, Y['model'] - hBH, LX, Y['softmax'] + hBH, C_IMG)
arrow_straight(ax, RX, Y['model'] - hBH, RX, Y['softmax'] + hBH, C_AUD)

# 模型 -> 物种路由 (虚线)
arrow_straight(ax, LX + BW/2, Y['model'], CX - 1.7, Y['model'], C_SPECIES, 1.3, True)
arrow_straight(ax, RX - BW/2, Y['model'], CX + 1.7, Y['model'], C_SPECIES, 1.3, True)

# 物种路由 -> Softmax (虚线)
arrow_straight(ax, CX - 1.0, Y['model'] - 0.75, LX + BW/2 - 0.2, Y['softmax'] + hBH,
               C_SPECIES, 1.3, True)
arrow_straight(ax, CX + 1.0, Y['model'] - 0.75, RX - BW/2 + 0.2, Y['softmax'] + hBH,
               C_SPECIES, 1.3, True)

# Softmax -> 映射
arrow_straight(ax, LX, Y['softmax'] - hBH, LX, Y['map'] + hBH, C_MAP)
arrow_straight(ax, RX, Y['softmax'] - hBH, RX, Y['map'] + hBH, C_MAP)

# 映射 -> UESS
arrow_straight(ax, LX, Y['map'] - hBH, LX, Y['uess'] + hBH, C_UESS)
arrow_straight(ax, RX, Y['map'] - hBH, RX, Y['uess'] + hBH, C_UESS)

# UESS -> 置信度
arrow_straight(ax, LX, Y['uess'] - hBH, LX, Y['conf'] + hBH, C_CONF)
arrow_straight(ax, RX, Y['uess'] - hBH, RX, Y['conf'] + hBH, C_CONF)

# 置信度 -> 权重
arrow_straight(ax, LX + 0.5, Y['conf'] - hBH, CX - 2.0, Y['weight'] + 0.525, C_WEIGHT)
arrow_straight(ax, RX - 0.5, Y['conf'] - hBH, CX + 2.0, Y['weight'] + 0.525, C_WEIGHT)

# UESS -> 融合 (投射分布直接进入融合)
# 左侧 u: 先垂直下到融合层高度，再水平到融合左端
ax.annotate('', xy=(CX - 2.9, Y['fusion']),
            xytext=(LX - 1.2, Y['uess'] - hBH),
            arrowprops=dict(arrowstyle='-|>,head_width=0.25,head_length=0.18',
                            lw=1.5, color=C_FUSION, linestyle='-',
                            connectionstyle='arc3,rad=0.12',
                            shrinkA=2, shrinkB=2),
            zorder=2)
# 右侧 v
ax.annotate('', xy=(CX + 2.9, Y['fusion']),
            xytext=(RX + 1.2, Y['uess'] - hBH),
            arrowprops=dict(arrowstyle='-|>,head_width=0.25,head_length=0.18',
                            lw=1.5, color=C_FUSION, linestyle='-',
                            connectionstyle='arc3,rad=-0.12',
                            shrinkA=2, shrinkB=2),
            zorder=2)

# 权重 -> 融合
arrow_straight(ax, CX, Y['weight'] - 0.525, CX, Y['fusion'] + 0.525, C_WEIGHT)

# 融合 -> Paining
arrow_straight(ax, CX, Y['fusion'] - 0.525, CX, Y['pain'] + 0.85, C_PAIN)

# 融合 -> 一致性 (虚线)
arrow_straight(ax, CX + 2.9, Y['fusion'], 15.8 - 1.5, Y['fusion'], C_CONSIST, 1.3, True)

# Paining -> 输出
arrow_straight(ax, CX, Y['pain'] - 0.85, CX, Y['output'] + 0.475, C_OUT)

# 一致性 -> 输出 (虚线弧线)
ax.annotate('', xy=(CX + 2.9, Y['output'] + 0.3),
            xytext=(15.8, Y['fusion'] - 0.475),
            arrowprops=dict(arrowstyle='-|>,head_width=0.2,head_length=0.15',
                            lw=1.2, color=C_CONSIST, linestyle=(0, (6, 4)),
                            connectionstyle='arc3,rad=-0.15',
                            shrinkA=2, shrinkB=2),
            zorder=2)

# 输出 -> 风险
arrow_straight(ax, CX, Y['output'] - 0.475, CX, Y['risk'] + 0.45, C_OUT)

# ============================================================
# 左侧阶段标注
# ============================================================
stages = [
    (Y['input'],   '输入层'),
    (Y['model'],   '模型推理'),
    (Y['softmax'], '概率输出'),
    (Y['map'],     '语义映射'),
    (Y['uess'],    '空间投射'),
    (Y['conf'],    '不确定性'),
    (Y['weight'],  '权重计算'),
    (Y['fusion'],  '融合决策'),
    (Y['pain'],    '安全覆盖'),
    (Y['output'],  '最终输出'),
    (Y['risk'],    '风险评估'),
]
for yy, txt in stages:
    stage_label(ax, yy, txt, x=1.3)

# 左侧引导线
for i in range(len(stages) - 1):
    y1 = stages[i][0] - 0.35
    y2 = stages[i+1][0] + 0.35
    ax.plot([1.3, 1.3], [y2, y1], color='#BDBDBD',
            linewidth=0.8, linestyle=':', zorder=1, alpha=0.5)

# ============================================================
# 保存
# ============================================================
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
output_path = f'd:/bcode/pet_result/figure/dsmf_flowchart_{timestamp}.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.3)
plt.close()
print(f'流程图已保存: {output_path}')
