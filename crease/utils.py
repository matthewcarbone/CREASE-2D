from time import perf_counter


class Timer:
    def __enter__(self):
        self._time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self._time = perf_counter() - self._time

    @property
    def elapsed(self):
        return self._time
