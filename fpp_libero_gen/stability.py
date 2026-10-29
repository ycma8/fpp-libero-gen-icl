"""Predeclared finite-instability detection; all ranks observe identical log rows."""
from collections import deque
import statistics


class FiniteInstabilityGuard:
    def __init__(self, after=5000, window=20, patience=3):
        self.after, self.window, self.patience = after, window, patience
        self.losses, self.grads = deque(maxlen=window), deque(maxlen=window)
        self.best = float('inf')
        self.bad = 0

    def state_dict(self):
        return dict(after=self.after, window=self.window, patience=self.patience,
                    losses=list(self.losses), grads=list(self.grads), best=self.best, bad=self.bad)

    def load_state_dict(self, state):
        if (state['after'], state['window'], state['patience']) != (self.after,self.window,self.patience):
            raise ValueError('Finite-instability guard configuration changed')
        self.losses.extend(state['losses']); self.grads.extend(state['grads'])
        self.best, self.bad = state['best'], state['bad']

    def observe(self, step, loss, grad):
        if step < self.after:
            return False
        self.losses.append(loss); self.grads.append(grad)
        if len(self.losses) < self.window:
            return False
        median = statistics.median(self.losses)
        self.best = min(self.best, median)
        unstable = median > max(.05, 5*self.best) and statistics.median(self.grads) > 10
        self.bad = self.bad+1 if unstable else 0
        return self.bad >= self.patience
