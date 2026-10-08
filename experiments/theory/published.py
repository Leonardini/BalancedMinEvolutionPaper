"""Numbers stated in the paper (from published.json, the single copy). They are used only
in asserted cross-checks, after the computed value has been written out, so a
disagreement stops the task with an error instead of passing silently."""
import json
from pathlib import Path

_P = json.loads((Path(__file__).resolve().parent / "published.json").read_text())


def _intkeys(d):
    return {int(k): v for k, v in d.items()}


CANONICAL_SURVIVORS = _intkeys(_P["canonical_survivors"])
ROW_TYPES = _intkeys(_P["row_types"])
CHERRY_FIXED_WITNESSES = _P["cherry_fixed_witnesses"]
CHERRY_FIXED_CLASSES = _P["cherry_fixed_classes"]
WEAK6_SURVIVORS = _P["weak6_survivors"]
WEAK6_NON_TREES = _P["weak6_non_trees"]
WEAK6_ORBITS = _P["weak6_orbits"]
WEAK6_STABILISER = _P["weak6_stabiliser"]
CHERRY_LIFT_VIOLATING_QUARTETS = _P["cherry_lift_violating_quartets"]
CHERRY_LIFT_QUARTET_1235_SUMS = _P["cherry_lift_quartet_1235_pair_sums"]
MEMBERSHIP_SYSTEMS = _intkeys(_P["membership_systems"])
MEMBERSHIP_N8_NO_TIEBREAK = _P["membership_n8_no_tiebreak"]
MIN_EDGE_CLIQUE_COVER = _intkeys(_P["min_edge_clique_cover"])
