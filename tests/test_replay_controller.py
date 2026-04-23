import unittest


class FakeReplayEngine:
    def __init__(self):
        self.initialized_with = None
        self.reset_with = None
        self.advanced_to = None

    def initialize(self, periods, start_time, max_count_map=None):
        self.initialized_with = (list(periods), start_time, dict(max_count_map or {}))

    def reset(self, start_time):
        self.reset_with = start_time

    def advance_to(self, end_time):
        self.advanced_to = end_time
        return end_time

    def get_view(self, period, count=300, with_indicators=True):
        return {"period": period, "count": count, "with_indicators": with_indicators}


class ReplayControllerTests(unittest.TestCase):
    def test_initialize_forwards_to_engine(self):
        from ui.controllers.replay_controller import ReplayController

        engine = FakeReplayEngine()
        controller = ReplayController(engine)

        controller.initialize(["1min"], "2026-04-24 09:30", {"1min": 800})

        self.assertEqual(
            engine.initialized_with,
            (["1min"], "2026-04-24 09:30", {"1min": 800}),
        )

    def test_advance_returns_actual_time_and_tracks_current_time(self):
        from ui.controllers.replay_controller import ReplayController

        engine = FakeReplayEngine()
        controller = ReplayController(engine)

        actual = controller.advance_to("2026-04-24 09:31")

        self.assertEqual(actual, "2026-04-24 09:31")
        self.assertEqual(controller.current_time, "2026-04-24 09:31")


if __name__ == "__main__":
    unittest.main()
