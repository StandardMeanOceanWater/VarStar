"""
VarStar Pipeline 流程圖 — 窄版（適合 A4 直式嵌入），中文大字
輸出：D:\VarStar\pipeline\pipeline_workflow_notitle.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["font.size"] = 11

# ── 畫布：窄版 ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 8.5))
ax.set_xlim(0, 9)
ax.set_ylim(0, 9)
ax.axis("off")

# ── 顏色 ─────────────────────────────────────────────
C_INPUT  = "#D6E4F0"
C_STAGE1 = "#EBF5FB"
C_STAGE2 = "#EAF4EA"
C_STAGE3 = "#FEF9E7"
C_OUTPUT = "#F5EEF8"
C_CONFIG = "#FDF2E9"
EDGE     = "#555555"

# ── 工具函式 ─────────────────────────────────────────
def box(ax, x, y, w, h, label, sublabel="", color="#FFF", fontsize=11, bold=False):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0.06",
        facecolor=color, edgecolor=EDGE, linewidth=1.1))
    weight = "bold" if bold else "normal"
    if sublabel:
        ax.text(x, y+0.15, label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color="#1a1a1a")
        ax.text(x, y-0.2,  sublabel, ha="center", va="center",
                fontsize=9, color="#555")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color="#1a1a1a")

def arrow(ax, x1, y1, x2, y2, dashed=False):
    ls = (0,(4,3)) if dashed else "solid"
    col = "#AAAAAA" if dashed else EDGE
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=1.3, linestyle=ls,
                                connectionstyle="arc3,rad=0.0"))

def bracket(ax, x, y1, y2, label, color):
    ax.plot([x,x],   [y1,y2], color=color, lw=2.8)
    ax.plot([x,x+0.12],[y1,y1], color=color, lw=2.8)
    ax.plot([x,x+0.12],[y2,y2], color=color, lw=2.8)
    ax.text(x-0.15, (y1+y2)/2, label, ha="right", va="center",
            fontsize=9.5, color=color, rotation=90, fontweight="bold")

# ── 節點座標 ─────────────────────────────────────────
BW, BH = 1.55, 0.82   # 方塊寬高
SW = 1.6

Y1, Y2, Y3, Y4 = 7.8, 5.5, 3.2, 1.1

# 第一列（前處理）
R1 = {"input":(0.95,Y1), "calib":(2.7,Y1), "solve":(4.45,Y1), "debayer":(6.2,Y1)}
# config 右側
cfg = (8.1, Y1)
# 第二列（測光）
R2 = {"comp":(1.8,Y2), "aperture":(3.6,Y2), "ensemble":(5.4,Y2), "timesync":(7.2,Y2)}
# 第三列（週期）
R3 = {"ls":(3.0,Y3), "fold":(5.0,Y3), "ratio":(7.0,Y3)}
# 輸出
OUT = {"csv":(3.8,Y4), "plot":(5.9,Y4), "log":(7.8,Y4)}

# ── 繪製節點 ─────────────────────────────────────────
box(ax, *R1["input"],   1.5,  BH, "原始影像", "CR2 / FITS",       C_INPUT,  bold=True)
box(ax, *R1["calib"],   BW,   BH, "Calibration", "Bias/Dark/Flat", C_STAGE1)
box(ax, *R1["solve"],   BW,   BH, "Plate Solve", "WCS 星圖解算",   C_STAGE1)
box(ax, *R1["debayer"], BW,   BH, "DeBayer",     "RGGB 四通道",    C_STAGE1)

box(ax, *R2["comp"],     SW,  BH, "比較星選取", "AAVSO/APASS/Gaia", C_STAGE2)
box(ax, *R2["aperture"], SW,  BH, "孔徑測光",   "生長曲線法",       C_STAGE2)
box(ax, *R2["ensemble"], SW,  BH, "差分校準",   "Ensemble 正規化",  C_STAGE2)
box(ax, *R2["timesync"], SW,  BH, "時間轉換",   "BJD_TDB",          C_STAGE2)

box(ax, *R3["ls"],   SW, BH, "頻譜分析", "Lomb-Scargle / DFT", C_STAGE3)
box(ax, *R3["fold"], SW, BH, "相位折疊", "Fourier 擬合",        C_STAGE3)
box(ax, *R3["ratio"],SW, BH, "G1/G2 比值","通率比值驗證",       C_STAGE3)

box(ax, *OUT["csv"],  1.4, 0.6, "測光數據 CSV", color=C_OUTPUT)
box(ax, *OUT["plot"], 1.6, 0.6, "3欄式分析圖",  color=C_OUTPUT, bold=True)
box(ax, *OUT["log"],  1.2, 0.6, "執行日誌",      color=C_OUTPUT)

# Config 虛線框
ax.add_patch(mpatches.FancyBboxPatch(
    (cfg[0]-0.8, cfg[1]-0.41), 1.6, 0.82,
    boxstyle="round,pad=0.06",
    facecolor=C_CONFIG, edgecolor=EDGE, linewidth=1.0, linestyle="dashed"))
ax.text(cfg[0], cfg[1]+0.13, "observation_config.yaml",
        ha="center", va="center", fontsize=8, style="italic", color="#555")
ax.text(cfg[0], cfg[1]-0.16, "全域參數設定",
        ha="center", va="center", fontsize=8.5, color="#777")

# ── 箭頭 ─────────────────────────────────────────────
# 第一列水平
for a_, b_ in [("input","calib"),("calib","solve"),("solve","debayer")]:
    x1 = R1[a_][0] + (0.75 if a_=="input" else BW/2)
    x2 = R1[b_][0] - BW/2
    arrow(ax, x1, Y1, x2, Y1)

# debayer ↓ comp
arrow(ax, R1["debayer"][0], Y1-BH/2, R2["comp"][0], Y2+BH/2)

# 第二列水平
for a_, b_ in [("comp","aperture"),("aperture","ensemble"),("ensemble","timesync")]:
    arrow(ax, R2[a_][0]+SW/2, Y2, R2[b_][0]-SW/2, Y2)

# timesync ↓ ls
arrow(ax, R2["timesync"][0], Y2-BH/2, R3["ls"][0], Y3+BH/2)

# 第三列水平
for a_, b_ in [("ls","fold"),("fold","ratio")]:
    arrow(ax, R3[a_][0]+SW/2, Y3, R3[b_][0]-SW/2, Y3)

# ratio ↓ 輸出
for key in ["csv","plot","log"]:
    ox, oy = OUT[key]
    arrow(ax, R3["ratio"][0], Y3-BH/2, ox, oy+0.3)

# config 虛線 → calib / solve / comp
for tx, ty in [(R1["calib"][0], Y1+BH/2),
               (R1["solve"][0],  Y1+BH/2),
               (R2["comp"][0],   Y2+BH/2)]:
    arrow(ax, cfg[0], cfg[1]-BH/2, tx, ty, dashed=True)

# ── 階段括號 ─────────────────────────────────────────
bracket(ax, 0.18, Y1-BH/2-0.05, Y1+BH/2+0.05, "階段一：前處理",  "#2471A3")
bracket(ax, 0.18, Y2-BH/2-0.05, Y2+BH/2+0.05, "階段二：核心測光","#1E8449")
bracket(ax, 0.18, Y3-BH/2-0.05, Y3+BH/2+0.05, "階段三：週期分析","#7D3C98")

# ── 輸出 ─────────────────────────────────────────────
out_path = r"D:\VarStar\pipeline\pipeline_workflow_notitle.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("已輸出：", out_path)
