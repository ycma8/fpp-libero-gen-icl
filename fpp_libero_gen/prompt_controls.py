"""In-distribution frame controls without changing the policy's random stream."""
import numpy as np


CONDITIONS = ('correct', 'wrong', 'blank', 'last_frame', 'shuffled')


def transform_demo(demo, condition, seed):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    order = np.arange(len(demo))
    if condition == 'last_frame':
        order[:] = len(demo)-1
    elif condition == 'shuffled':
        order = np.random.default_rng(seed + 1000003).permutation(len(demo))
        if len(order) > 1 and np.array_equal(order, np.arange(len(order))):
            order = np.roll(order, 1)
    result = np.asarray(demo[order]).copy()
    if condition == 'blank':
        result.fill(0)
    return result, order.tolist()
