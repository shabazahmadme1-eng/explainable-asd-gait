# -*- coding: utf-8 -*-
"""Figures for the Neuroscience Informatics (NSI) v2 package, full-length version.

Fig. 1  Kinect v2 25-joint skeleton in NTU order with the bone graph from backend/config.py,
        marking the flexion-angle vertices, the torso length and the four normalized distances.
Fig. 2  study pipeline (American spelling; CV scope drawn explicitly).
Fig. 3  stream ablation: per-seed child-level accuracy from v40_artifacts/v40_3seed_summary.json.
Fig. 4  child-level confusion matrices of the only two pipelines run under the identical
        three-seed protocol (averaged OOF counts from the original package's Fig. 7).
Fig. 5  ten largest EBM term importances, read from colab/nsi_model_facts.json.
Graphical abstract.

Current NSI artwork rules: vector drawings as EPS/PDF with embedded fonts; bitmapped line
drawings as TIFF/JPG/PNG at >= 1000 dpi. Each figure is written as a vector PDF (TrueType
fonts embedded), a 1000-dpi LZW TIFF and a PNG (the PNG is only what gets embedded in the Word
file; the build asserts its pixels equal the TIFF's).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
plt.rcParams.update({'font.family': 'Arial', 'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42})
OUT = os.path.join(HERE, 'figs_nsi_v2')
os.makedirs(OUT, exist_ok=True)
DPI = 1000  # current NSI guide: bitmapped line drawings >= 1000 dpi (vector PDF preferred)

LEFT, RIGHT, MID = '#2F7FBF', '#D9731A', '#4A4F55'
HC_C, EMB_C, FUS_C = '#DD6B1F', '#2E9B55', '#7A57D1'


def save(fig, stem):
    png = os.path.join(OUT, stem + '.png')
    fig.savefig(png, dpi=DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(OUT, stem + '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    im = Image.open(png).convert('RGB')
    im.save(os.path.join(OUT, stem + '.tif'), compression='tiff_lzw', dpi=(DPI, DPI))
    print(stem, im.size)


# ---------------------------------------------------------------- Fig. 1 --- #
NAMES = ['SpineBase', 'SpineMid', 'Neck', 'Head', 'ShoulderLeft', 'ElbowLeft', 'WristLeft', 'HandLeft',
         'HandTipLeft', 'ThumbLeft', 'ShoulderRight', 'ElbowRight', 'WristRight', 'HandRight', 'HandTipRight',
         'ThumbRight', 'HipLeft', 'KneeLeft', 'AnkleLeft', 'FootLeft', 'HipRight', 'KneeRight', 'AnkleRight',
         'FootRight', 'SpineShoulder']
PARENTS = [0, 0, 24, 2, 24, 4, 5, 6, 7, 6, 24, 10, 11, 12, 13, 12, 0, 16, 17, 18, 0, 20, 21, 22, 1]
# Front view with the subject facing the reader: the subject's left side is drawn on the reader's right.
HALF = {'SpineBase': (0, 0.95), 'SpineMid': (0, 1.22), 'SpineShoulder': (0, 1.47), 'Neck': (0, 1.56),
        'Head': (0, 1.72), 'ShoulderLeft': (0.20, 1.44), 'ElbowLeft': (0.29, 1.17), 'WristLeft': (0.35, 0.93),
        'HandLeft': (0.37, 0.86), 'HandTipLeft': (0.39, 0.78), 'ThumbLeft': (0.31, 0.85),
        'HipLeft': (0.10, 0.92), 'KneeLeft': (0.12, 0.52), 'AnkleLeft': (0.13, 0.11), 'FootLeft': (0.20, 0.05)}
pos = {}
for n, (x, y) in HALF.items():
    pos[n] = (x, y)
    if n.endswith('Left'):
        pos[n.replace('Left', 'Right')] = (-x, y)
assert sorted(pos) == sorted(NAMES)
side = lambda n: LEFT if n.endswith('Left') else RIGHT if n.endswith('Right') else MID

fig, ax = plt.subplots(figsize=(4.6, 5.4))
for i, p in enumerate(PARENTS):
    if i != p:
        (x1, y1), (x2, y2) = pos[NAMES[i]], pos[NAMES[p]]
        ax.plot([x1, x2], [y1, y2], color='#B9BEC4', lw=3.2, solid_capstyle='round', zorder=1)
for a, b in [('WristLeft', 'WristRight'), ('AnkleLeft', 'AnkleRight'), ('Head', 'WristLeft'), ('Head', 'WristRight')]:
    (x1, y1), (x2, y2) = pos[a], pos[b]
    ax.plot([x1, x2], [y1, y2], color='#7D848B', lw=0.9, ls=(0, (3, 2.5)), zorder=0)
for n in NAMES:
    ax.scatter(*pos[n], s=46, color=side(n), edgecolor='white', lw=0.8, zorder=3)
for n in ['ElbowLeft', 'ElbowRight', 'KneeLeft', 'KneeRight', 'ShoulderLeft', 'ShoulderRight', 'HipLeft', 'HipRight']:
    ax.scatter(*pos[n], s=190, facecolor='none', edgecolor='#1B2226', lw=1.1, zorder=4)
ax.annotate('', xy=(0.58, 0.95), xytext=(0.58, 1.47),
            arrowprops=dict(arrowstyle='<->', color='#1B2226', lw=1.0, shrinkA=0, shrinkB=0))
for y in (0.95, 1.47):
    ax.plot([0.03, 0.56], [y, y], color='#7D848B', lw=0.6, ls=':', zorder=0)
# Spine joints are named in the torso note, not beside the midline, where labels collide with the rings.
ax.text(0.62, 1.21, 'torso length\n(SpineShoulder to\nSpineBase; distance\nnormalization)', fontsize=8,
        va='center', color='#1B2226')
ax.text(pos['Head'][0] + 0.05, pos['Head'][1], 'Head', fontsize=7.5, ha='left', va='center', color=MID)
ax.text(-0.26, 1.83, "Subject's right", fontsize=8.5, ha='center', color=RIGHT, fontweight='bold')
ax.text(0.26, 1.83, "Subject's left", fontsize=8.5, ha='center', color=LEFT, fontweight='bold')
handles = [Line2D([], [], marker='o', ls='', color=LEFT, label='Left-side joint'),
           Line2D([], [], marker='o', ls='', color=RIGHT, label='Right-side joint'),
           Line2D([], [], marker='o', ls='', color=MID, label='Midline joint'),
           Line2D([], [], marker='o', ls='', mfc='none', mec='#1B2226', ms=10, label='Flexion-angle vertex'),
           Line2D([], [], color='#7D848B', ls=(0, (3, 2.5)), label='Normalized distance')]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=8, frameon=False)
ax.set_xlim(-0.62, 0.98); ax.set_ylim(-0.02, 1.92); ax.set_aspect('equal'); ax.axis('off')
save(fig, 'Figure_1')

# ---------------------------------------------------------------- Fig. 2 --- #
fig, ax = plt.subplots(figsize=(6.3, 7.6))
ax.set_xlim(0, 10); ax.set_ylim(0.9, 12.2); ax.axis('off')

INPUT = ('#dbe8fb', '#3b6fd4'); PRE = ('#e5e9ef', '#5f6b7a')
LRN = ('#dcf3e3', '#2e9b55'); LRN2 = ('#eefaf1', '#2e9b55')
HC = ('#fde7d3', '#dd6b1f'); HC2 = ('#fff4ea', '#dd6b1f')
AGG = ('#dbe8fb', '#3b6fd4'); FUS = ('#ece6fb', '#7a57d1')
OUTP = ('#fdf1cf', '#c98a0b'); EBM = ('#fbe3ef', '#c7337a')


def box(cx, cy, w, h, text, style, fs=9.6, bold=False):
    fc, ec = style
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle='round,pad=0.03,rounding_size=0.12',
                                fc=fc, ec=ec, lw=1.3))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, color='#161616',
            fontweight='bold' if bold else 'normal', linespacing=1.3)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=11,
                                 color='#2f3640', lw=1.2, shrinkA=0, shrinkB=0))


box(5, 11.62, 4.6, 0.85, '3D skeletal gait\n(25-joint coordinates)', INPUT, fs=10.2, bold=True)
arrow(5, 11.19, 5, 10.80)
box(5, 10.36, 6.6, 0.85, 'Pre-processing\ninterpolate · median filter · re-center · resample', PRE)

ax.add_patch(Rectangle((0.25, 3.02), 9.5, 6.72, fill=False, ec='#8c949c', lw=1.0, ls=(0, (5, 3))))
# Two lines at the top-left corner so the label clears the arrows leaving pre-processing.
ax.text(0.42, 9.64, 'Child-grouped\n5-fold CV × 3 seeds', fontsize=8.4, style='italic', color='#6b737b',
        va='top', linespacing=1.2)

arrow(5, 9.93, 2.62, 9.05); arrow(5, 9.93, 7.38, 9.05)
box(2.55, 8.35, 4.3, 1.30, 'Learned representation\nTwo-stream MS-G3D (joint + bone)\n768-D embedding → PCA 150', LRN)
box(7.45, 8.35, 4.3, 1.30, 'Biomechanical representation\n534 interpretable features\nfold-wise filtering', HC)
arrow(2.55, 7.70, 2.55, 7.22); arrow(7.45, 7.70, 7.45, 7.22)
box(2.55, 6.62, 4.3, 1.12, 'Regularized tree ensemble\nXGB / LGB / CatBoost\n→ $P_{\\mathrm{EMB}}$', LRN2)
box(7.45, 6.62, 4.3, 1.12, 'Regularized tree ensemble\nXGB / LGB / CatBoost\n→ $P_{\\mathrm{HC}}$', HC2)
arrow(2.55, 6.06, 4.2, 5.36); arrow(7.45, 6.06, 5.8, 5.36)
box(5, 4.95, 6.6, 0.80, 'Child-level aggregation\n(mean over the clips of each child)', AGG)
arrow(5, 4.55, 5, 4.18)
box(5, 3.62, 7.2, 1.05, 'Decision-level fusion\n$p_{\\mathrm{final}} = w\\,P_{\\mathrm{EMB}} + (1-w)\\,P_{\\mathrm{HC}}$\nWeight selected on scored OOF outcomes (not nested)',
    FUS, fs=8.5, bold=True)
arrow(5, 3.095, 2.75, 2.40); arrow(5, 3.095, 7.25, 2.40)
box(2.55, 1.88, 4.3, 0.95, 'Predictive output\nchild-level probability / label', OUTP)
box(7.45, 1.88, 4.3, 0.95, 'EBM surrogate\nbiomechanical approximation', EBM)
save(fig, 'Figure_2')

# ---------------------------------------------------------------- Fig. 3 --- #
summ = json.load(open(os.path.join(ROOT, 'v40_artifacts', 'v40_3seed_summary.json')))
SEEDS = [42, 123, 2024]
hc = [100 * np.mean([f['hc_subj_acc'] for f in summ['fold_details'] if f['seed'] == s]) for s in SEEDS]
emb = [100 * np.mean([f['emb_subj_acc'] for f in summ['fold_details'] if f['seed'] == s]) for s in SEEDS]
fus = [95.0, 92.0, 93.0]  # per-seed fused accuracy from the training log (per-fold fused values were not kept)
assert np.allclose(hc, [87, 82, 85]) and np.allclose(emb, [94, 91, 93]), (hc, emb)
fig, ax = plt.subplots(figsize=(5.4, 3.9))
MARK = ['o', 's', '^']
for i, (lab, vals, c) in enumerate([('Handcrafted\nstream', hc, HC_C), ('MS-G3D\nstream', emb, EMB_C),
                                    ('Decision-level\nfusion', fus, FUS_C)]):
    m, sd = np.mean(vals), np.std(vals)  # ddof = 0, as in the saved summary
    ax.bar(i, m, width=0.58, color=c, alpha=0.22, edgecolor=c, lw=1.3, zorder=1)
    ax.errorbar(i, m, yerr=sd, color='#1B2226', capsize=5, lw=1.2, zorder=2)
    # Seed markers sit right of the error bar so neither hides the other; the label clears both.
    for j, v in enumerate(vals):
        ax.scatter(i + 0.10 + 0.08 * j, v, marker=MARK[j], s=38, color=c, edgecolor='#1B2226', lw=0.7, zorder=3)
    ax.text(i, max(max(vals), m + sd) + 0.9, '%.1f ± %.1f' % (m, sd), ha='center', fontsize=9.5)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['Handcrafted\nstream', 'MS-G3D\nstream', 'Decision-level\nfusion'])
ax.set_ylim(75, 100); ax.set_ylabel('Child-level accuracy (%)')
ax.yaxis.grid(True, color='#E3E6E8', lw=0.8); ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(handles=[Line2D([], [], marker=MARK[j], ls='', color='#9AA1A8', mec='#1B2226', label='Seed %d' % SEEDS[j])
                   for j in range(3)], loc='lower right', fontsize=8.5, frameon=False)
save(fig, 'Figure_3')

# ---------------------------------------------------------------- Fig. 4 --- #
# Averaged-OOF child-level matrices [[TN, FP], [FN, TP]] (v1 Fig. 7, bottom row).
PANELS = [('GCN+Transformer backbone', [[48, 2], [6, 44]], '92.0', '0.981'),
          ('MS-G3D backbone (proposed)', [[49, 1], [4, 46]], '95.0', '0.986')]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
for ax, (title, m, acc, auc) in zip(axes, PANELS):
    ax.imshow(m, cmap='Greys', vmin=0, vmax=50)
    for i in range(2):
        for j in range(2):
            v = m[i][j]
            ax.text(j, i, '%s\n%d' % (['TN', 'FP', 'FN', 'TP'][2 * i + j], v), ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white' if v > 25 else 'black')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['TD', 'ASD'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['TD', 'ASD'])
    ax.set_xlabel('Predicted', fontsize=9.5); ax.set_ylabel('True', fontsize=9.5)
    ax.set_title('%s\naccuracy %s%%, AUC %s' % (title, acc, auc), fontsize=9.5)
    ax.tick_params(labelsize=9.5, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
fig.tight_layout(w_pad=2.5)
save(fig, 'Figure_4')

# ---------------------------------------------------------------- Fig. 5 --- #
facts = json.load(open(os.path.join(HERE, 'nsi_model_facts.json')))
LABEL = {'bio_ankle_dist_norm_t1': 'Inter-ankle distance (t1)',
         'bio_R_shoulder_angle_t3': 'Right shoulder flexion (t3)',
         'bio_R_shoulder_angle_t5': 'Right shoulder flexion (t5)',
         'bio_R_elbow_angle_t2': 'Right elbow flexion (t2)',
         'bio_L_hip_angle_t6': 'Left knee flexion (t6)',       # legacy name: vertex at the knee
         'frame_asym_angle_std': 'SD of frame-wise angular asymmetry',
         'bio_R_elbow_angle_t3': 'Right elbow flexion (t3)',
         'tau_sym_elbow': 'Bilateral elbow synchrony (Kendall’s tau)',
         'bio_R_elbow_angle_t0': 'Right elbow flexion (t0)',
         'std_R_shoulder_angle': 'SD of right shoulder flexion'}
top = facts['top_terms'][:10]
assert [t['name'] for t in top] == list(LABEL), [t['name'] for t in top]
fig, ax = plt.subplots(figsize=(6.2, 3.9))
ys = np.arange(len(top))[::-1]
for y, t in zip(ys, top):
    sampled = t['name'].startswith('bio_')
    ax.barh(y, t['importance'], height=0.62, color='#3B6FD4' if sampled else '#C7337A')
    ax.text(t['importance'] + 0.0006, y, '%.3f' % t['importance'], va='center', fontsize=8.5)
ax.set_yticks(ys); ax.set_yticklabels([LABEL[t['name']] for t in top], fontsize=9)
ax.set_xlabel('Mean absolute contribution to the fused probability (unsigned)', fontsize=9)
ax.set_xlim(0, 0.048)
ax.xaxis.grid(True, color='#E3E6E8', lw=0.8); ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(handles=[Rectangle((0, 0), 1, 1, color='#3B6FD4', label='Single sampled frame'),
                   Rectangle((0, 0), 1, 1, color='#C7337A', label='Clip-level summary')],
          loc='lower right', fontsize=8.5, frameon=False)
save(fig, 'Figure_5')

# ------------------------------------------------------ graphical abstract --- #
fig, ax = plt.subplots(figsize=(10, 4.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis('off')
ax.text(5, 3.95, 'Skeletal-gait fusion for autism-related motor atypicality', ha='center', fontsize=17, weight='bold')
items = [(.25, 'DATA', '50 ASD + 50 TD\n800 cached clips\n25 Kinect joints'),
         (3.55, 'TWO REPRESENTATIONS', 'MS-G3D embeddings\nBiomechanical descriptors\nDecision-level fusion'),
         (6.85, 'INTERNAL VALIDATION', '3 × child-grouped 5-fold CV\nAccuracy 93.3 ± 1.2%\nAUC 0.979 ± 0.007')]
for x, title, body in items:
    ax.add_patch(FancyBboxPatch((x, 1.45), 2.9, 1.85, boxstyle='round,pad=.08', fc='#edf2f7', ec='#64748b'))
    ax.text(x + 1.45, 2.96, title, ha='center', fontsize=10, weight='bold')
    ax.text(x + 1.45, 2.15, body, ha='center', va='center', fontsize=11, linespacing=1.6)
for x in [3.2, 6.5]:
    ax.annotate('', xy=(x + .27, 2.35), xytext=(x, 2.35), arrowprops={'arrowstyle': '->', 'lw': 2})
ax.text(5, .95, 'EBM surrogate: in-sample r = 0.959  |  Mean fusion increment: 0.7 points', ha='center', fontsize=11)
ax.text(5, .43, 'OOF weight selection introduces optimism; external validation remains necessary.', ha='center',
        fontsize=10, color='#475569')
save(fig, 'Graphical_Abstract')
