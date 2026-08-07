import app


def test_audit_status_indicator_handles_nan_without_crashing():
    # overall_score/audit_status come back as float NaN (not None) once a
    # DataFrame column mixes real strings with SQL NULLs. This used to crash
    # with AttributeError: 'float' object has no attribute 'lower'.
    assert app.audit_status_indicator(float("nan")) == "⚪"


def test_audit_status_indicator_handles_none():
    assert app.audit_status_indicator(None) == "⚪"


def test_audit_status_indicator_handles_empty_string():
    assert app.audit_status_indicator("") == "⚪"


def test_audit_status_indicator_maps_verified_to_green():
    assert app.audit_status_indicator("Verified (Active Queue)") == "🟢"


def test_audit_status_indicator_maps_needs_attention_to_yellow():
    assert app.audit_status_indicator("Needs Attention") == "🟡"


def test_audit_status_indicator_maps_disqualified_to_red():
    assert app.audit_status_indicator("Disqualified Archive") == "🔴"


def test_score_indicator_handles_nan_without_crashing():
    # Unscored leads carry NaN, not None, once loaded into a mixed-dtype
    # pandas DataFrame. Unlike audit_status_indicator, this function already
    # guarded against it via try/except — this test locks that in.
    assert app.score_indicator(float("nan")) == "⚪"


def test_score_indicator_handles_none():
    assert app.score_indicator(None) == "⚪"


def test_score_indicator_maps_high_score_to_green():
    assert app.score_indicator(95) == "🟢"


def test_score_indicator_maps_mid_score_to_yellow():
    assert app.score_indicator(75) == "🟡"


def test_score_indicator_maps_low_score_to_red():
    assert app.score_indicator(50) == "🔴"
