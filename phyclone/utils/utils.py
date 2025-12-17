import time
from collections import deque
from itertools import count


def get_iterator_length(iterable):
    counter = count()
    deque(zip(iterable, counter), maxlen=0)
    return next(counter)


class Timer:
    """Taken from https://www.safaribooksonline.com/library/view/python-cookbook-3rd/9781449357337/ch13s13.html"""

    def __init__(self, func=time.time):
        self.elapsed = 0.0

        self._func = func

        self._start = None

    @property
    def running(self):
        return self._start is not None

    def reset(self):
        self.elapsed = 0.0

    def start(self):
        if self._start is not None:
            raise RuntimeError("Already started")

        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError("Not started")

        end = self._func()

        self.elapsed += end - self._start

        self._start = None

    def __enter__(self):
        self.start()

        return self

    def __exit__(self, *args):
        self.stop()


class TraceEntry:
    __slots__ = "iter", "time", "alpha", "log_p_one", "tree", "tree_hash"

    def __init__(self, i, timer, tree, tree_dist):
        self.iter = i
        self.time = timer.elapsed
        self.alpha = tree_dist.prior.alpha
        self.log_p_one = tree_dist.log_p_one(tree)
        self.tree = tree.to_storage_tree()
        self.tree_hash = tree.get_hash_id_obj()
