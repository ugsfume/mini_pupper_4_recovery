"""``--widen-limits-deg``: hip and knee targets past the track ends, recorded in the bounds so an evaluation keeps it."""
import numpy as np

from mp4_recovery import limits as L


def test_widening_moves_only_the_track_margin():
    plain, wide = L.decoder_bounds("getup"), L.decoder_bounds("getup", widen_limits_deg=60.0)
    assert plain.info["track_margin_deg"] == L.TRACK_MARGIN_DEG and plain.info["widen_limits_deg"] is None
    assert wide.info["track_margin_deg"] == -60.0 and wide.info["widen_limits_deg"] == 60.0
    assert np.all(wide.lo <= plain.lo) and np.all(wide.hi >= plain.hi)
    assert np.any(wide.lo < plain.lo) and np.any(wide.hi > plain.hi)
    assert np.array_equal(wide.calf_lo, plain.calf_lo) and np.array_equal(wide.calf_hi, plain.calf_hi)


def test_evaluation_reads_the_widening_from_the_meta():
    meta = {"limits": True, "bounds": L.decoder_bounds("getup", widen_limits_deg=60.0).to_meta()}
    assert L.eval_plant(meta)["widen_limits_deg"] == 60.0
    assert L.eval_plant({"limits": True, "bounds": L.decoder_bounds("getup").to_meta()})["widen_limits_deg"] is None
