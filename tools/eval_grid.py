#!/usr/bin/env python
"""
Run the SwinJSCC checkpoints over the (SNR, CBR) grids used in the paper and
write one long-form CSV that tools/plot_rd.py turns into RD curves.

Design notes:
  * main.py is NOT modified. Its `config` class is imported (with a patched
    sys.argv) and reused, so dataset paths and the encoder/decoder kwargs keep
    exactly one definition -- this script cannot drift from main.py.
  * Metrics use utils.AverageMeter exactly like main.py: PSNR is taken from the
    batch-mean MSE (not per-image PSNR), so the numbers are protocol-identical
    to the authors' test().
  * Each run pairs ONE checkpoint with the SNR and bottleneck dimension C it
    was trained for. For `SwinJSCC_w/o_SAandRA` the head dimension is baked
    into the weights, so C must match or load_state_dict fails on a size
    mismatch -- that failure is loud, not silent.
  * For `SwinJSCC_w/o_SAandRA` the paper draws one point per separately trained
    model (p.9), which is why that variant needs 5 checkpoints for one curve.

Usage:
    python tools/eval_grid.py --list                    # run table, no eval
    python tools/eval_grid.py --figs fig11b --limit 2   # quick smoke test
    python tools/eval_grid.py                           # default AWGN+CIFAR10
    python tools/eval_grid.py --figs fig10e fig11e      # Rayleigh figures
"""

import argparse
import csv
import importlib
import math
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from loss.distortion import MS_SSIM
from utils import AverageMeter

# ---------------------------------------------------------------- model names

WO = 'SwinJSCC_w/o_SAandRA'
SA = 'SwinJSCC_w/_SA'
RA = 'SwinJSCC_w/_RA'
SARA = 'SwinJSCC_w/_SAandRA'

SNRS = [1, 4, 7, 10, 13]
CS = [32, 64, 96, 128, 192]  # CBR 1/48, 1/24, 1/16, 1/12, 1/8 (downsample=4)
C_CBR1_16 = 96
CKPT = 'checkpoint'
KNOWN_FIGS = ['fig10a', 'fig10c', 'fig10e', 'fig11b', 'fig11e', 'fig13b', 'fig13d']
DEFAULT_FIGS = ['fig10c', 'fig11b', 'fig13b', 'fig13d', 'fig10a']


def cbr_of(c, downsample):
    """CBR = C / (2 * 3 * 2^(2*downsample)), the paper's formula (p.95)."""
    return c / (2.0 * 3.0 * 2 ** (2 * downsample))


def as_list(x):
    return x if isinstance(x, list) else [x]


# ------------------------------------------------------------------- run table

def build_runs():
    """Flat list of runs; the figure grouping lives in the `fig` column, so
    adding a figure here is all that is needed."""
    runs = []

    def add(fig, ckpt, model, channel, metric, trainset, testset, size, C, snr):
        runs.append(dict(fig=fig, ckpt=os.path.join(CKPT, ckpt), model=model,
                         channel=channel, metric=metric, trainset=trainset,
                         testset=testset, size=size, C=C, snr=snr))

    # --- Fig.10 (c)/(e): PSNR vs SNR at average CBR 1/16 -------------------
    # Panel letters follow the PAPER: (c) is Kodak AWGN, (e) is Kodak Rayleigh.
    # ((d) is CIFAR10 Rayleigh, which needs CIFAR10 weights + a different C.)
    for fig, ch, tag in (('fig10c', 'awgn', 'AWGN'),
                         ('fig10e', 'rayleigh', 'Rayleigh')):
        for s in SNRS:
            add(fig, 'SwinJSCC_wo_SAandRA_%s_HRimage_snr%d_psnr_C96.model' % (tag, s),
                WO, ch, 'MSE', 'DIV2K', 'kodak', 'base', C_CBR1_16, s)
        add(fig, 'SwinJSCC_w_SA_%s_HRimage_snr_psnr_C96.model' % tag,
            SA, ch, 'MSE', 'DIV2K', 'kodak', 'base', C_CBR1_16, SNRS)
        add(fig, 'SwinJSCC_w_SAandRA_%s_HRimage_cbr_psnr_snr.model' % tag,
            SARA, ch, 'MSE', 'DIV2K', 'kodak', 'base', C_CBR1_16, SNRS)

    # --- Fig.11 (b)/(e): PSNR vs CBR at SNR=10dB (AWGN) / 3dB (Rayleigh) ---
    for fig, ch, tag, s0, wo_tag in (
            ('fig11b', 'awgn', 'AWGN', 10, 'AWGN'),
            ('fig11e', 'rayleigh', 'Rayleigh', 3, 'Rayleigh')):
        for c in CS:
            add(fig, 'SwinJSCC_wo_SAandRA_%s_HRimage_snr%d_psnr_C%d.model' % (wo_tag, s0, c),
                WO, ch, 'MSE', 'DIV2K', 'kodak', 'base', c, s0)
        ra_ck = ('SwinJSCC_w_RA_AWGN_HRimage_cbr_psnr_snr10.model' if ch == 'awgn'
                 else 'SwinJSCC_w_RA_Raylegh_HRimage_cbr_psnr_snr3_v2.model')
        add(fig, ra_ck, RA, ch, 'MSE', 'DIV2K', 'kodak', 'base', CS, s0)
        add(fig, 'SwinJSCC_w_SAandRA_%s_HRimage_cbr_psnr_snr.model' % tag,
            SARA, ch, 'MSE', 'DIV2K', 'kodak', 'base', CS, s0)

    # --- Fig.13 (b): MS-SSIM vs SNR at average CBR 1/16 (AWGN only -- the
    #     released weights contain no Rayleigh MS-SSIM models) --------------
    for s in SNRS:
        add('fig13b', 'SwinJSCC_wo_SAandRA_AWGN_HRimage_snr%d_msssim_C96.model' % s,
            WO, 'awgn', 'MS-SSIM', 'DIV2K', 'kodak', 'base', C_CBR1_16, s)
    add('fig13b', 'SwinJSCC_w_SA_AWGN_HRimage_snr_msssim_C96.model',
        SA, 'awgn', 'MS-SSIM', 'DIV2K', 'kodak', 'base', C_CBR1_16, SNRS)
    add('fig13b', 'SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_msssim_snr.model',
        SARA, 'awgn', 'MS-SSIM', 'DIV2K', 'kodak', 'base', C_CBR1_16, SNRS)

    # --- Fig.13 (d): MS-SSIM vs CBR at SNR = 10dB (AWGN) ------------------
    for c in CS:
        add('fig13d', 'SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_msssim_C%d.model' % c,
            WO, 'awgn', 'MS-SSIM', 'DIV2K', 'kodak', 'base', c, 10)
    add('fig13d', 'SwinJSCC_w_RA_AWGN_HRimage_cbr_msssim_snr10.model',
        RA, 'awgn', 'MS-SSIM', 'DIV2K', 'kodak', 'base', CS, 10)
    add('fig13d', 'SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_msssim_snr.model',
        SARA, 'awgn', 'MS-SSIM', 'DIV2K', 'kodak', 'base', CS, 10)

    # --- Fig.10 (a): CIFAR10, PSNR vs SNR at average CBR 1/3 --------------
    # Only w/ SA was released for CIFAR10; there is no w/o SA&RA CIFAR10
    # checkpoint, so the paper's w/o curve cannot be reproduced.
    add('fig10a', 'SwinJSCC_w_SA_AWGN_CIFAR10_snr_psnr_C32.model',
        SA, 'awgn', 'MSE', 'CIFAR10', 'kodak', 'base', 32, SNRS)
    add('fig10a', 'SwinJSCC_w_SA_Rayleigh_CIFAR10_snr_psnr_C32.model',
        SA, 'rayleigh', 'MSE', 'CIFAR10', 'kodak', 'base', 32, SNRS)

    return runs


# ------------------------------------------------------- reuse main.py config

_MAIN_CACHE = {}


def get_main(model_size, trainset, testset):
    """Import main.py with a patched argv so its `config` class is built for
    the requested setup. Cached per setup; reloaded when the setup changes."""
    key = (model_size, trainset, testset)
    if key in _MAIN_CACHE:
        return _MAIN_CACHE[key]
    saved = sys.argv
    sys.argv = ['eval_grid.py', '--model_size', model_size,
                '--trainset', trainset, '--testset', testset]
    try:
        if 'main' in sys.modules:
            m = importlib.reload(sys.modules['main'])
        else:
            import main as m
    finally:
        sys.argv = saved
    _MAIN_CACHE[key] = m
    return m


def configure(main_mod, run):
    """Return (args, config, snr_list, c_list) for one run.

    NOTE: SwinJSCC.__init__ parses `args.multiple_snr` and `args.C` with
    .split(',') followed by int(), so both must be COMMA STRINGS -- handing it
    a Python list makes int('[32') blow up.
    """
    snr_list = as_list(run['snr'])
    c_list = as_list(run['C'])

    cfg = main_mod.config
    cfg.encoder_kwargs = dict(cfg.encoder_kwargs)
    cfg.decoder_kwargs = dict(cfg.decoder_kwargs)
    # w/o SA&RA and w/_SA bake the bottleneck dimension into the weights; the
    # adaptive variants take it at run time and use C=None.
    c_in_model = c_list[0] if run['model'] in (WO, SA) else None
    for kw in (cfg.encoder_kwargs, cfg.decoder_kwargs):
        kw['model'] = run['model']
        kw['C'] = c_in_model

    args = argparse.Namespace(**vars(main_mod.args))
    args.model = run['model']
    args.C = ','.join(str(c) for c in c_list)
    args.channel_type = run['channel']
    args.distortion_metric = run['metric']
    args.multiple_snr = ','.join(str(s) for s in snr_list)
    return args, cfg, snr_list, c_list


def make_test_loader(cfg, trainset):
    if trainset == 'CIFAR10':
        from torchvision import datasets as tvd
        from torchvision import transforms
        ds = tvd.CIFAR10(root=cfg.test_data_dir, train=False,
                         transform=transforms.ToTensor(), download=False)
        return torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
    from data.datasets import Datasets
    return torch.utils.data.DataLoader(Datasets(cfg.test_data_dir),
                                       batch_size=1, shuffle=False)


def evaluate(net, loader, snr, rate, calcu_ssim):
    """Same accumulation as main.py's test(): AverageMeter over batches."""
    psnrs, msssims, cbrs = AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        for batch in loader:
            img = batch[0].cuda()
            recon, CBR, _snr, mse, _ = net(img, snr, rate)
            cbrs.update(float(CBR))
            if mse.item() > 0:
                psnrs.update((10 * (torch.log(255. * 255. / mse) / np.log(10))).item())
                msssims.update(1.0 - calcu_ssim(img, recon.clamp(0., 1.)).mean().item())
    if psnrs.count == 0:
        return None
    ms = msssims.avg
    return dict(CBR=cbrs.avg, psnr=psnrs.avg, msssim=ms,
                msssim_db=-10.0 * math.log10(max(1e-12, 1.0 - ms)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--figs', nargs='*', default=None,
                    help='figure groups to run (default: %s)' % ' '.join(DEFAULT_FIGS))
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--out', default=None, help='csv path (default results/rd_grid.csv)')
    ap.add_argument('--limit', type=int, default=0, help='run only the first N runs')
    ap.add_argument('--list', action='store_true', help='print the run table and exit')
    a = ap.parse_args()

    figs = a.figs if a.figs is not None else DEFAULT_FIGS
    unknown = [f for f in figs if f not in KNOWN_FIGS]
    if unknown:
        print('unknown figure(s): %s\nknown: %s' % (unknown, KNOWN_FIGS))
        return 1

    runs = [r for r in build_runs() if r['fig'] in figs]
    if not runs:
        print('no runs selected')
        return 1

    # ---- preflight: report missing checkpoints instead of crashing later ---
    missing = sorted({r['ckpt'] for r in runs if not os.path.exists(r['ckpt'])})
    if missing:
        print('!! %d checkpoint(s) not found (those runs are skipped):' % len(missing))
        for p in missing:
            print('   %s' % p)
        print('')
    runs = [r for r in runs if os.path.exists(r['ckpt'])]
    if not runs:
        print('nothing left to run')
        return 1

    if a.list:
        print('%-8s %-22s %-9s %-8s %-6s %s' %
              ('fig', 'model', 'channel', 'metric', 'C', 'snr'))
        for r in runs:
            print('%-8s %-22s %-9s %-8s %-6s %s' %
                  (r['fig'], r['model'], r['channel'], r['metric'],
                   ','.join(str(x) for x in as_list(r['C'])),
                   ','.join(str(x) for x in as_list(r['snr']))))
        print('\n%d runs over %d checkpoints'
              % (len(runs), len({r['ckpt'] for r in runs})))
        return 0

    if a.limit:
        runs = runs[:a.limit]

    os.makedirs(a.results_dir, exist_ok=True)
    out_path = a.out or os.path.join(a.results_dir, 'rd_grid.csv')

    # ---- group by setup so main.py is reloaded once per setup -------------
    groups = []
    for r in runs:
        key = (r['size'], r['trainset'], r['testset'])
        for g in groups:
            if g[0] == key:
                g[1].append(r)
                break
        else:
            groups.append((key, [r]))

    rows = []
    for (size, trainset, testset), grp in groups:
        main_mod = get_main(size, trainset, testset)
        loader = make_test_loader(main_mod.config, trainset)
        downsample = 2 if trainset == 'CIFAR10' else 4
        calcu_ssim = (MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
                      if trainset == 'CIFAR10'
                      else MS_SSIM(data_range=1., levels=4, channel=3).cuda())

        for r in grp:
            args, cfg, snr_list, c_list = configure(main_mod, r)
            print('== %s | %s | %s | C=%s | snr=%s'
                  % (r['fig'], r['model'], r['channel'],
                     ','.join(str(c) for c in c_list),
                     ','.join(str(s) for s in snr_list)))
            print('   %s' % os.path.basename(r['ckpt']))

            net = main_mod.SwinJSCC(args, cfg)
            sd = torch.load(r['ckpt'], map_location='cpu')
            net.load_state_dict(sd, strict=True)
            net = net.cuda().eval()
            del sd

            for snr in snr_list:
                for rate in c_list:
                    res = evaluate(net, loader, snr, rate, calcu_ssim)
                    if res is None:
                        print('   snr=%-3s C=%-4s -> no valid mse, skipped' % (snr, rate))
                        continue
                    rows.append(dict(
                        fig=r['fig'], model=r['model'], channel=r['channel'],
                        distortion=r['metric'], trainset=trainset, testset=testset,
                        C=rate, CBR=round(cbr_of(rate, downsample), 6), snr=snr,
                        psnr=round(res['psnr'], 4),
                        msssim=round(res['msssim'], 6),
                        msssim_db=round(res['msssim_db'], 4),
                        checkpoint=os.path.basename(r['ckpt'])))
                    print('   snr=%-3s C=%-4s CBR=%.4f  PSNR=%.3f  MS-SSIM=%.4f (%.2f dB)'
                          % (snr, rate, res['CBR'], res['psnr'],
                             res['msssim'], res['msssim_db']))

            del net
            torch.cuda.empty_cache()

    cols = ['fig', 'model', 'channel', 'distortion', 'trainset', 'testset', 'C',
            'CBR', 'snr', 'psnr', 'msssim', 'msssim_db', 'checkpoint']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print('\nwrote %d rows -> %s' % (len(rows), out_path))
    return 0


if __name__ == '__main__':
    sys.exit(main())
