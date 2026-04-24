class ReplayController:
    def __init__(self, replay_engine):
        self.replay_engine = replay_engine
        self.enabled = False
        self.is_playing = False
        self.speed = 60
        self.current_time = None

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)
        if not self.enabled:
            self.is_playing = False

    def set_speed(self, speed: int):
        self.speed = int(speed)

    def toggle_playing(self):
        self.is_playing = not self.is_playing
        return self.is_playing

    def initialize(self, periods, start_time, max_count_map=None):
        self.current_time = start_time
        self.replay_engine.initialize(periods, start_time, max_count_map=max_count_map)

    def reset(self, start_time):
        self.current_time = start_time
        self.replay_engine.reset(start_time)

    def advance_to(self, end_time):
        actual_time = self.replay_engine.advance_to(end_time)
        if actual_time is not None:
            self.current_time = actual_time
        return actual_time

    def get_view(self, period, count=300, with_indicators=True):
        return self.replay_engine.get_view(period, count=count, with_indicators=with_indicators)
