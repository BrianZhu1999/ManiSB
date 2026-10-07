"""重画用户截图中的原始平滑闭合曲线案例。"""
from pathlib import Path
import csv
import hashlib
import json
import socket
import sys

if socket.gethostname().lower().replace('_', '-') != 'super-server':
    raise RuntimeError('绘图与图像检查只能在 super-server 执行')

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE.parent/'multibranch_figure_20261002/qa_libs'))
import pymupdf as fitz

OUT = BASE/'output'
W, H = 340., 90.
METHODS = ['triad', 'corrected']
LABELS = ['No correction', 'Learned\ncorrection']
COLORS = ['#72808E', '#8670AD', '#2A938D']
FILLS = ['#CBD1D6', '#D4C9E4', '#AFD7D2']
BLUE = '#0072B2'
INK = '#000000'

mpl.rcParams.update({
    'font.family':'Arial', 'font.sans-serif':['Arial'], 'font.size':10,
    'text.color':INK, 'axes.labelcolor':INK, 'xtick.color':INK, 'ytick.color':INK,
    'axes.titlesize':12, 'axes.titleweight':'normal', 'axes.labelsize':10,
    'xtick.labelsize':10, 'ytick.labelsize':10, 'legend.fontsize':10,
    'axes.linewidth':.55, 'font.weight':'normal', 'mathtext.fontset':'custom',
    'mathtext.rm':'Arial', 'mathtext.it':'Arial:italic', 'mathtext.bf':'Arial:bold',
    'mathtext.sf':'Arial', 'mathtext.fallback':None,
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
    'savefig.facecolor':'white', 'figure.facecolor':'white',
})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(v, pos=None):
    return '0' if abs(v)<1e-12 else f'{v:g}'


def text(fig, x, y, value, size=10, **kwargs):
    return fig.text(x/W, y/H, value, fontsize=size, color=INK, va='center', **kwargs)


def header(fig, x, letter, title):
    text(fig, x, 83.5, letter, 14, weight='bold')
    text(fig, x+5.5, 83.5, title, 12)


def axes(fig, x, y, w, h):
    return fig.add_axes([x/W, y/H, w/W, h/H])


def clean_geo(ax):
    ax.set(xlim=(-1.35, 1.35), ylim=(-1.4, 1.2), aspect='equal')
    ax.axis('off')


def curve(ax, xy, **kwargs):
    ax.plot(xy[:,0], xy[:,1], solid_capstyle='round', **kwargs)


def arrow(ax, start, stop, color=INK, width=.75):
    ax.annotate('', xy=stop, xytext=start,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=width,
                                mutation_scale=6, shrinkA=0, shrinkB=0))


def draw_task(fig, z):
    ax = axes(fig, 3, 16, 66, 62)
    clean_geo(ax)
    curve(ax, z['target_curve'], color=BLUE, lw=1.2)
    curve(ax, z['terminal_curve'], color='#7B848C', lw=.8, ls=(0,(2.3,2)))
    ids = z['selected_ids'][::2]
    for j in ids:
        a, b = z['terminal'][j], z['target'][j]
        ax.plot([a[0],b[0]], [a[1],b[1]], color=BLUE, lw=.65, alpha=.38)
        arrow(ax, a*.58+b*.42, a*.44+b*.56, color=BLUE, width=.65)
    ax.scatter(z['terminal'][ids,0], z['terminal'][ids,1], s=9, color=INK, zorder=5)
    ax.scatter(z['target'][ids,0], z['target'][ids,1], s=12, color=BLUE, zorder=5)
    text(fig, 37, 73.5, r'$X_1\;\rightarrow\;X_0$', ha='center')
    legend = axes(fig, 7, 5, 64, 8)
    legend.axis('off')
    legend.legend(handles=[Line2D([0],[0],color=BLUE,lw=1.2,label=r'Target $X_0$'),
                           Line2D([0],[0],color='#7B848C',lw=.8,ls=(0,(2.3,2)),label=r'Input $X_1$')],
                  loc='center', ncol=2, frameon=False, handlelength=1.7, columnspacing=1.5)


def draw_paths(fig, z):
    ax = axes(fig, 77, 16, 72, 62)
    clean_geo(ax)
    curve(ax, z['target_curve'], color='#AAB6C0', lw=.85)
    curve(ax, z['terminal_curve'], color='#AAB6C0', lw=.7, ls=(0,(2.3,2)))
    ids = z['selected_ids']
    for method, col, style in [('triad',COLORS[1],(0,(3,2))), ('corrected',COLORS[2],'-')]:
        trajectory = z['trajectory_'+method]
        ax.add_collection(LineCollection(trajectory[:,ids].transpose(1,0,2), colors=col,
                          linewidths=.95, linestyles=style, alpha=.92))
        ax.scatter(trajectory[0,ids,0], trajectory[0,ids,1], s=7, color=INK, zorder=5)
        ax.scatter(trajectory[-1,ids,0], trajectory[-1,ids,1], s=10, color=col,
                   edgecolor='white', linewidth=.25, zorder=5)
    for j in ids[[1,5]]:
        traj = z['trajectory_corrected']
        arrow(ax, traj[25,j], traj[29,j], color=COLORS[2])
    detail = axes(fig, 155, 37, 34, 34)
    j = int(z['detail_sample_id'][0])
    k = 30
    support = z['support_curves'][k]
    tree = cKDTree(support)
    selected = []
    for method, col, style in [('triad',COLORS[1],(0,(3,2))), ('corrected',COLORS[2],'-')]:
        pt = z['trajectory_'+method][k,j]
        _, idx = tree.query(pt)
        foot = support[idx]
        selected.extend([pt,foot])
        detail.plot([pt[0],foot[0]], [pt[1],foot[1]], color=col, lw=.85, ls=style)
        detail.scatter(*pt, s=17, color=col, edgecolor='white', linewidth=.35, zorder=5)
        detail.scatter(*foot, s=12, facecolor='white', edgecolor=col, linewidth=.8, zorder=5)
    values = np.asarray(selected)
    center = values.mean(axis=0)
    span = max(np.ptp(values[:,0]),np.ptp(values[:,1]),.045)*1.7
    detail.set_xlim(center[0]-span/2,center[0]+span/2)
    detail.set_ylim(center[1]-span/2,center[1]+span/2)
    curve(detail, support, color='#8E9AA5', lw=.8)
    detail.set_aspect('equal')
    detail.set_xticks([])
    detail.set_yticks([])
    for spine in detail.spines.values():
        spine.set_linewidth(.55)
        spine.set_color('#737B83')
    rect = Rectangle((center[0]-span/2,center[1]-span/2),span,span,fill=False,ec='#737B83',lw=.55)
    ax.add_patch(rect)
    fig.add_artist(ConnectionPatch(xyA=(center[0]+span/2,center[1]), coordsA=ax.transData,
                   xyB=(0,.5), coordsB=detail.transAxes, color='#8E9AA5', lw=.5, clip_on=False))
    text(fig, 172, 74.2, r'$t=0.5$', ha='center')
    text(fig, 172, 31, 'Distance to support', ha='center')
    legend = axes(fig, 78, 4.5, 110, 9)
    legend.axis('off')
    legend.legend(handles=[Line2D([0],[0], color=COLORS[1], lw=1, ls=(0,(3,2)), label='No correction'),
                           Line2D([0],[0], color=COLORS[2], lw=1, label='Learned correction')],
                  loc='center', ncol=2, frameon=False, handlelength=2, columnspacing=1.6)
    return dict(detail_sample_id=j, detail_time=float(z['times'][k]), detail_extent=[*detail.get_xlim(), *detail.get_ylim()])


def draw_metric(fig, datasets, metric, x):
    ax = axes(fig, x, 20, 55, 54)
    all_values = [np.concatenate([z[metric+'_'+name] for z in datasets]) for name in METHODS]
    artist = ax.boxplot(all_values, positions=[1,2], widths=.48, whis=(5,95),
                  showfliers=True, patch_artist=True,
                  medianprops=dict(color=INK,lw=.8),
                  flierprops=dict(marker='.',markersize=1.2,markeredgewidth=0,
                                  markerfacecolor='#87939F',alpha=.16))
    for i, box in enumerate(artist['boxes']):
        box.set(facecolor=FILLS[i+1], edgecolor=COLORS[i+1], linewidth=.7)
        for item in artist['whiskers'][2*i:2*i+2]+artist['caps'][2*i:2*i+2]:
            item.set(color=COLORS[i+1],lw=.65)
    for i,name in enumerate(METHODS):
        means = [z[metric+'_'+name].mean() for z in datasets]
        ax.scatter([i+1], means, marker='D', s=17,
                   facecolor='white',edgecolor=INK,linewidth=.45,zorder=5)
    ax.set_xticks([1,2], LABELS)
    ax.set_xlim(.4,2.6)
    upper = max(float(v.max()) for v in all_values)*1.07
    ax.set_ylim(0,upper)
    ticks = MaxNLocator(nbins=5,min_n_ticks=4).tick_values(0,upper)
    ax.set_yticks(ticks[(ticks>=0)&(ticks<=upper)])
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.spines[['right','top']].set_visible(False)
    ax.tick_params(axis='both',width=.55,length=2.3,pad=3)
    ax.set_ylabel('Euclidean distance' if metric=='endpoint' else 'Mean distance')
    return {'metric':metric, 'scale':ax.get_yscale(), 'limits':list(ax.get_ylim()),
            'n_values':len(all_values[0]), 'all_values_within_axes':all(v.min()>=0 and v.max()<upper for v in all_values)}


def main():
    OUT.mkdir(exist_ok=True)
    report = json.loads((BASE/'data/data_audit.json').read_text())
    path = BASE/'data/closed_coupling_data.npz'
    assert sha(path)==report['data_sha256']
    datasets = [dict(np.load(path))]
    fig = plt.figure(figsize=(W/25.4,H/25.4),dpi=150)
    header(fig,4,'a','Smooth closed coupling')
    header(fig,77,'b','Learned path correction')
    header(fig,199,'c','Endpoint distance')
    header(fig,274,'d','Path-support distance')
    draw_task(fig,datasets[0])
    detail = draw_paths(fig,datasets[0])
    quantitative = [draw_metric(fig,datasets,'endpoint',207),draw_metric(fig,datasets,'support',282)]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    text_issues, typography, visible = [], [], []
    for obj in fig.findobj(Text):
        if not obj.get_visible() or not obj.get_text():
            continue
        extent = obj.get_window_extent(renderer)
        if extent.width==0 or extent.height==0:
            continue
        if not (bounds.x0-.5<=extent.x0 and bounds.y0-.5<=extent.y0 and extent.x1<=bounds.x1+.5 and extent.y1<=bounds.y1+.5):
            text_issues.append(obj.get_text())
        is_panel = obj.get_text() in ['a','b','c','d']
        if obj.get_fontsize() not in [10,12,14] or (obj.get_fontweight()=='bold' and not is_panel):
            typography.append(obj.get_text())
        if mpl.colors.to_hex(obj.get_color()) != INK:
            typography.append('nonblack: '+obj.get_text())
        visible.append((obj.get_text(),extent))
    collisions=[]
    for i,(name,a) in enumerate(visible):
        for name2,b in visible[i+1:]:
            # 重复的不可见对侧刻度在 Matplotlib 中共享内容，忽略同一位置的同文本副本。
            if name==name2 and np.allclose(a.bounds,b.bounds):
                continue
            ix=min(a.x1,b.x1)-max(a.x0,b.x0)
            iy=min(a.y1,b.y1)-max(a.y0,b.y0)
            if ix>1 and iy>1:
                collisions.append([name,name2])
    prefix=OUT/'fig_smooth_closed_coupling'
    for suffix in ['pdf','svg']:
        fig.savefig(prefix.with_suffix('.'+suffix))
    fig.savefig(prefix.with_suffix('.png'),dpi=600)
    fig.savefig(prefix.with_suffix('.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(OUT/'preview.png',dpi=180)
    doc=fitz.open(prefix.with_suffix('.pdf'))
    doc[0].get_pixmap(dpi=180).save(OUT/'pdf_preview.png')
    Image.open(OUT/'preview.png').convert('L').save(OUT/'grayscale_preview.png')
    metrics=[dict(method=name,**report['metrics'][name]) for name in METHODS]
    with (OUT/'plotted_metrics.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    qa=dict(canvas_mm=[W,H],source_sha256=sha(path),training_seed=1234,evaluation_seed=20261002,
            detail=detail,metric_axes=quantitative,text_outside_canvas=text_issues,
            typography_violations=typography,text_collisions=collisions,
            parameters=report['parameters'],
            statistics='Same 1024 endpoint pairs and same initial noise. Boxes describe samples from one existing checkpoint; diamonds mark means.',
            full_range=True,aggregate=report['metrics'],
            discretization=report['discretization'],
            fonts=[list(f) for f in doc[0].get_fonts()])
    (OUT/'figure_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps(qa,indent=2))
    if text_issues or typography or collisions:
        raise RuntimeError('文字边界或字号检查未通过')


if __name__=='__main__':
    main()
