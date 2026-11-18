from app.worker.runner import stage_plan


def test_plan_sums_to_100_without_diarization():
    plan = stage_plan({"diarize": False})
    assert "diarize" not in [name for name, _ in plan]
    assert abs(sum(w for _, w in plan) - 100) < 1e-6


def test_plan_sums_to_100_with_diarization():
    plan = stage_plan({"diarize": True})
    assert "diarize" in [name for name, _ in plan]
    assert abs(sum(w for _, w in plan) - 100) < 1e-6