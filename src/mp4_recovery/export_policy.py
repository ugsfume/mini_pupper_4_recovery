#!/usr/bin/env python
"""Export a training checkpoint as a robot policy directory: ``policy.npz`` (numpy = torch to 1e-5) and the run's options
in ``config.json``.

    python -m mp4_recovery.export_policy runs/getup/<run>/model_4546.pt policies/mine

The joint limits and the stance offset come from the run's options (``limits``, ``stand_offset_abd_deg``) unless
``--limits`` / ``--no-limits`` / ``--stand-offset-abd`` say otherwise; with limits the meta gets the decoder's clamps
(``limits.decoder_bounds(kind)``, with the run's ``widen_limits_deg``). ``--from-policy DIR``: no checkpoint: a copy of an
exported policy with the limits and offset written into its meta (a stand change without retraining):

    python -m mp4_recovery.export_policy --from-policy policies/walker_base policies/walker --limits --stand-offset-abd 12

The target filter's alpha (meta ``action_filter_alpha``) is the run's option ``filter_alpha`` (the source policy's meta with
``--from-policy``; 0 when absent) unless ``--filter-alpha`` says otherwise.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from mp4_recovery.policy_file import export, with_meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="a checkpoint (or with --from-policy: the source policy directory)")
    ap.add_argument("out")
    ap.add_argument("--from-policy", action="store_true",
                    help="CKPT is an exported policy directory: copy it with --limits / --stand-offset-abd in the meta")
    ap.add_argument("--limits", action="store_true", default=None,
                    help="the decoder's clamps in the meta (default: the run's option 'limits')")
    ap.add_argument("--no-limits", dest="limits", action="store_false")
    ap.add_argument("--stand-offset-abd", type=float, metavar="DEG",
                    help="the abductions' stand offset, deg outward (default: the run's option, else 0)")
    ap.add_argument("--filter-alpha", type=float, metavar="ALPHA",
                    help="the decoder's target filter alpha (default: the run's option 'filter_alpha', else 0)")
    ap.add_argument("--kind", choices=("walk", "getup"),
                    help="the policy's kind (default: the run's option 'kind', else walk)")
    ap.add_argument("--band-deg", type=float, nargs="+", metavar="DEG",
                    help="the action band about the stand: 1, 3 (abduction, hip, knee) or 12 values (default: the "
                         "run's option 'band_deg', else the walking band 20/60/60)")
    a = ap.parse_args()
    from mp4_recovery.limits import decoder_bounds
    ck, out = Path(a.ckpt), Path(a.out)
    if a.from_policy:
        meta = json.loads(str(np.load(ck / "policy.npz")["meta"]))
        kind = a.kind or meta.get("kind") or "walk"
        lim = bool(meta.get("limits")) if a.limits is None else a.limits
        off = float(meta.get("stand_offset_abd_deg", 0.0) or 0.0) if a.stand_offset_abd is None else a.stand_offset_abd
        alpha = float(meta.get("action_filter_alpha", 0.0) or 0.0) if a.filter_alpha is None else a.filter_alpha
        widen = ((meta.get("bounds") or {}).get("info") or {}).get("widen_limits_deg")
        bounds = decoder_bounds(kind, widen_limits_deg=widen).to_meta() if lim else None
        path = with_meta(ck, out, kind=kind, limits=lim, bounds=bounds,
                         stand_offset_abd_deg=float(off), action_filter_alpha=float(alpha))
        if (ck / "config.json").exists():
            shutil.copy(ck / "config.json", out / "config.json")
        print("EXPORTED", path)
        return
    cfg = json.loads((ck.parent / "config.json").read_text())
    opts = cfg["task"]["options"]
    kind = a.kind or opts.get("kind") or "walk"
    lim = bool(opts.get("limits", False)) if a.limits is None else a.limits
    off = float(opts.get("stand_offset_abd_deg", 0.0) or 0.0) if a.stand_offset_abd is None else a.stand_offset_abd
    alpha = float(opts.get("filter_alpha", 0.0) or 0.0) if a.filter_alpha is None else a.filter_alpha
    bounds = decoder_bounds(kind, widen_limits_deg=opts.get("widen_limits_deg")).to_meta() if lim else None
    path = export(ck, out, opts["action_scale"], history=int(opts.get("history", 0) or 0), band_deg=a.band_deg if a.band_deg is not None else opts.get("band_deg"),
                  kind=kind, getup_phase_s=float(opts.get("phase_s", 0.0) or 0.0),
                  bounds=bounds, stand_offset_abd_deg=off, action_filter_alpha=alpha)
    # only what evaluation needs: a run's config also names local paths
    (out / "config.json").write_text(json.dumps({"task": cfg["task"], "noise": cfg.get("noise", "measured")}, indent=1))
    print("EXPORTED", path)


if __name__ == "__main__":
    main()
