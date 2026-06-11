import random
import time
import winsound

class TouchGame:
    def __init__(self, duration=5):

        self.duration = duration

        self.targets = [
            "NOSE",
            "LEFT EYE",
            "RIGHT EYE",
            "LIPS",
            "LEFT EAR",
            "RIGHT EAR",
            "CHIN",
            "FOREHEAD",
            "LEFT SHOULDER",
            "RIGHT SHOULDER"
        ]

        self.current_target = random.choice(self.targets)
        self.target_time = time.time()

        self.scores = { "Player 1": 0,"Player 2": 0 }

        self.round_winner = None

    def update_target(self):

        if time.time() - self.target_time > self.duration:
            self.next_round()

    def next_round(self):

        self.current_target = random.choice(
            [t for t in self.targets if t != self.current_target]
        )
        self.target_time = time.time()
        self.round_winner = None

    def check_winner(self, player_name, detections):

        # prevent multiple scoring in same round
        if self.round_winner is not None:
            return False

        if self.current_target in detections:

            self.scores[player_name] += 1
            self.round_winner = player_name
            winsound.Beep(1000, 200)

            return True

        return False