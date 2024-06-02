from metrics import MetricScore, args

class Experiments:
    def __init__(self):
        self.metrics = MetricScore()

    def run(self):
        self.metrics.print_score()

if __name__ == "__main__":
    experiments = Experiments()
    experiments.run()
