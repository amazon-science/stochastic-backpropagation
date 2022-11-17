class LossMeter(object):

    def __init__(self):
        self.total = 0
        self.count = 0

    def update(self, val, num=1):
        self.total += val
        self.count += num

    def compute(self):
        return self.total / self.count

    def reset(self):
        self.total = 0
        self.count = 0
