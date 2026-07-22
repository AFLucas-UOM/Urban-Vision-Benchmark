from run_followup_experiments import PILOT_MODELS, resolution_decision


def _metrics(map_value, small_ap, small_ar):
    return {"map50_95": map_value, "map_small": small_ap, "ar_small": small_ar}


def test_resolution_decision_requires_mean_small_gain_and_map_guardrail():
    baseline = {model: _metrics(.50, .10, .20) for model in PILOT_MODELS}
    improved = {model: _metrics(.50, .125, .21) for model in PILOT_MODELS}
    assert resolution_decision(baseline, improved)["proceed_to_1280"]
    no_small_gain = {model: _metrics(.52, .11, .21) for model in PILOT_MODELS}
    assert not resolution_decision(baseline, no_small_gain)["proceed_to_1280"]
    map_regression = {model: _metrics(.48, .13, .23) for model in PILOT_MODELS}
    assert not resolution_decision(baseline, map_regression)["proceed_to_1280"]
