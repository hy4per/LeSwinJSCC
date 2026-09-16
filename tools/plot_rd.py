#!/usr/bin/env python
"""
Turn results/rd_grid.csv (written by tools/eval_grid.py) into the RD-curve
figures used in the paper.

Each figure group from the CSV becomes one plot per channel, styled to match
the paper: same axis meanings, same per-method colours and markers:
    SwinJSCC w/o SA&RA  blue, '*'      SwinJSCC w/ SA&RA  red, '*'
    SwinJSCC w/ SA      cyan, 's'      SwinJSCC w/ RA     cyan, 's'
The paper reuses cyan for whichever adaptive variant appears in a given panel
(it never shows w/ SA and w/ RA in the same panel). If our data ever does put
both in one axes the script moves w/ RA to orange so the lines stay readable.

`--paper-refs` overlays the published curves as hollow open markers so that a
real discrepancy is visible at a glance. Where our reproduction is correct the
overlay hides exactly behind our own filled markers. See PAPER_REFS for the
provenance of each set: fig10a was vector-extracted from the PDF and is
essentially exact; the others were eyeballed and are only good to ~0.3 dB.

Usage:
    python tools/plot_rd.py                       # all figures present in the csv
    python tools/plot_rd.py --paper-refs          # with the paper overlay
    python tools/plot_rd.py --csv results/rd_grid.csv --outdir results/figs
"""

import argparse
import csv
import collections
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WO = 'SwinJSCC_w/o_SAandRA'
SA = 'SwinJSCC_w/_SA'
RA = 'SwinJSCC_w/_RA'
SARA = 'SwinJSCC_w/_SAandRA'

STYLE = {
    WO:   dict(color='blue', marker='*', label='SwinJSCC w/o SA&RA'),
    SARA: dict(color='red',  marker='*', label='SwinJSCC w/ SA&RA'),
    SA:   dict(color='c',    marker='s', label='SwinJSCC w/ SA'),
    RA:   dict(color='c',    marker='s', label='SwinJSCC w/ RA'),
}
CONFLICT_COLOR = 'darkorange'

# x/y column and axis labels per figure group
FIG_META = {
    'fig10a': dict(x='snr',       y='psnr',       xlabel='SNR (dB)',               ylabel='PSNR (dB)'),
    'fig10c': dict(x='snr',       y='psnr',       xlabel='SNR (dB)',               ylabel='PSNR (dB)'),
    # Panel letters follow the paper: (c) Kodak AWGN, (e) Kodak Rayleigh,
    # (d) would be CIFAR10 Rayleigh.
    'fig10e': dict(x='snr',       y='psnr',       xlabel='SNR (dB)',               ylabel='PSNR (dB)'),
    'fig11b': dict(x='CBR',       y='psnr',       xlabel='channel bandwidth ratio', ylabel='PSNR (dB)'),
    'fig11e': dict(x='CBR',       y='psnr',       xlabel='channel bandwidth ratio', ylabel='PSNR (dB)'),
    'fig13b': dict(x='snr',       y='msssim_db',  xlabel='SNR (dB)',               ylabel='MS-SSIM (dB)'),
    'fig13d': dict(x='CBR',       y='msssim_db',  xlabel='channel bandwidth ratio', ylabel='MS-SSIM (dB)'),
}

# Reference values from the published figures. PROVENANCE MATTERS HERE:
#
#   fig10a/c, fig11b -- VECTOR-EXTRACTED from the PDF (tools/extract_paper_refs.py,
#                       gridline-calibrated), good to ~0.01 dB. Trustworthy.
#   fig13b           -- EYEBALLED from a 7x render, only good to ~0.3 dB. I got
#                       CIFAR10 wrong twice this way. Rough sanity check only.
#
# The extractor is NOT trustworthy for the BPG+*/DeepJSCC baselines: those
# series pick up the axes frame and must be read from the rendered figure.
# See the note in tools/extract_paper_refs.py:curve_points.
#
# Where our reproduction is correct the overlay hides exactly behind our own
# markers and is invisible -- the overlay only earns its keep by making a real
# discrepancy obvious.
PAPER_REFS = {
    ('fig10a', 'awgn'): {
        SA: [(1, 29.535), (4, 32.58), (7, 35.34), (10, 37.74), (13, 39.680)],
    },
    ('fig10c', 'awgn'): {
        WO:   [(1, 29.7540), (4, 31.2772), (7, 32.6846),
               (10, 33.6947), (13, 34.4849)],
        SA:   [(1, 29.6598), (4, 31.2986), (7, 32.6219),
               (10, 33.6044), (13, 34.2923)],
        SARA: [(1, 29.5283), (4, 31.0636), (7, 32.2617),
               (10, 33.1756), (13, 33.8025)],
    },
    ('fig11b', 'awgn'): {
        WO:   [(0.020833, 29.3267), (0.041667, 31.8820), (0.0625, 33.6045),
               (0.083333, 35.0028), (0.125, 37.1397)],
        RA:   [(0.020833, 29.0491), (0.041667, 31.5766), (0.0625, 33.2286),
               (0.083333, 34.4528), (0.125, 36.0577)],
        SARA: [(0.020833, 29.0337), (0.041667, 31.5531), (0.0625, 33.1755),
               (0.083333, 34.4021), (0.125, 36.0427)],
    },
    ('fig13b', 'awgn'): {
        SA:   [(1, 13.4), (4, 15.7), (7, 17.7), (10, 19.3), (13, 20.7)],
        SARA: [(1, 13.3), (4, 15.5), (7, 17.2), (10, 18.6), (13, 19.5)],
    },
}


def dataset_title(rows):
    r = rows[0]
    if r['trainset'] == 'CIFAR10':
        return 'CIFAR10'
    return r['testset'].capitalize()


CHANNEL_TITLE = {'awgn': 'AWGN', 'rayleigh': 'Rayleigh'}


def channel_title(channel):
    return CHANNEL_TITLE.get(channel.lower(), channel)


def load_rows(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['snr'] = float(r['snr'])
        r['CBR'] = float(r['CBR'])
        r['psnr'] = float(r['psnr'])
        r['msssim_db'] = float(r['msssim_db'])
    return rows


def plot_panel(ax, fig, channel, rows, meta, paper_refs=False):
    xk, yk = meta['x'], meta['y']
    models = sorted({r['model'] for r in rows}, key=lambda m: [WO, SA, RA, SARA].index(m)
                    if m in (WO, SA, RA, SARA) else 99)
    both_adaptive = SA in models and RA in models

    for m in models:
        pts = sorted(((r[xk], r[yk]) for r in rows if r['model'] == m))
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        st = STYLE.get(m, dict(color='green', marker='o', label=m))
        color = st['color']
        if both_adaptive and m == RA:
            color = CONFLICT_COLOR
        ax.plot(xs, ys, marker=st['marker'], color=color, lw=1.6, ms=7,
                label=st.get('label', m), zorder=3)

    if paper_refs:
        refs = PAPER_REFS.get((fig, channel), {})
        for m, pts in refs.items():
            st = STYLE.get(m, {})
            c = st.get('color', 'grey')
            if both_adaptive and m == RA:
                c = CONFLICT_COLOR
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls='none', marker='o',
                    mfc='none', mec=c, mew=1.2, ms=11, alpha=0.55, zorder=2,
                    label='_nolegend_')

    ax.set_xlabel(meta['xlabel'])
    ax.set_ylabel(meta['ylabel'])
    ax.grid(True, which='both', alpha=0.4)
    ax.set_title('%s dataset, %s channel' % (dataset_title(rows), channel_title(channel)))
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.tick_params(labelsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', default=os.path.join(REPO_ROOT, 'results', 'rd_grid.csv'))
    ap.add_argument('--outdir', default=os.path.join(REPO_ROOT, 'results', 'figs'))
    ap.add_argument('--paper-refs', action='store_true',
                    help='overlay values read off the published figures (approximate)')
    a = ap.parse_args()

    if not os.path.exists(a.csv):
        print('csv not found: %s' % a.csv)
        return 1
    rows = load_rows(a.csv)
    if not rows:
        print('csv is empty')
        return 1

    groups = collections.OrderedDict()
    for r in rows:
        groups.setdefault((r['fig'], r['channel']), []).append(r)

    os.makedirs(a.outdir, exist_ok=True)
    made = []
    for (fig, channel), grp in sorted(groups.items()):
        meta = FIG_META.get(fig)
        if meta is None:
            print('skipping unknown figure group %r (%d rows)' % (fig, len(grp)))
            continue
        plt.figure(figsize=(5.2, 4.0))
        ax = plt.gca()
        plot_panel(ax, fig, channel, grp, meta, a.paper_refs)
        plt.tight_layout()
        out = os.path.join(a.outdir, 'rd_%s_%s%s.png'
                           % (fig, channel, '_withrefs' if a.paper_refs else ''))
        plt.savefig(out, dpi=160)
        plt.close()
        made.append((out, len(grp), meta['x'], meta['y']))
        print('wrote %s  (%d rows, x=%s y=%s)' % (out, len(grp), meta['x'], meta['y']))

    if not made:
        print('nothing plotted')
        return 1
    print('\n%d figure(s) -> %s' % (len(made), a.outdir))
    return 0


if __name__ == '__main__':
    sys.exit(main())
