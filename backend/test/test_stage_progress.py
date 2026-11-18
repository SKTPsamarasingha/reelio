from app.worker.runner import StageProgress, stage_plan


class Recorder:
    def __init__(self):
        self.values = []

    def report(self, stage, progress, message="", force=False):
        self.values.append(progress)


def test_progress_is_monotonic_and_reaches_100():
    rec = Recorder()
    sp = StageProgress(rec, {"diarize": False})
    for name, _ in stage_plan({"diarize": False}):
        sp.begin(name)
        sp.at(0.5)
        sp.at(0.2)                 # smaller value must be ignored
        sp.end()
    assert rec.values == sorted(rec.values)
    assert abs(rec.values[-1] - 100) < 1e-6


def test_parts_split_one_stage():
    rec = Recorder()
    sp = StageProgress(rec, {"diarize": False})
    sp.begin("cut", parts=2)
    sp.at(1.0)
    first = rec.values[-1]
    sp.next_part()
    sp.at(1.0)
    assert rec.values[-1] > first