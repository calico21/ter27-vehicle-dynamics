from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar, Any, Optional

T = TypeVar('T')

class TimeWindowedQueue(Generic[T]):
    """Ordered Queue of timed entries that keeps entries that fall in a given time window before the entry with the latest time"""

    @dataclass
    class Entry(Generic[T]):
        time_ns: int
        content: T

    def __init__(self, time_window_ns: int):
        self._queue: deque[TimeWindowedQueue.Entry] = deque()
        self._time_window_ns = time_window_ns

    def insert(self, time_ns: int, content: T):
        insert_index = self.get_earliest_index_later_than(time_ns)
        self._queue.insert(insert_index, TimeWindowedQueue.Entry(time_ns, content))
        self._clean()

    def get_earliest_index_later_than(self, time_ns: int, later_or_equal: bool = False) -> int:
        """
        Returns the index of the earliest element that occured later than time_ns. 
        If later_or_equal flag is true, returns the index of the earliest element that occured either later or at the same time than time_ns
        """
        for i in range(len(self._queue)):
            if time_ns < self._queue[i].time_ns or (time_ns == self._queue[i].time_ns and later_or_equal):
                return i
        return len(self._queue)

    def __len__(self):
        return self._queue.__len__()
    
    def __getitem__(self, key) -> Entry[T]:
        return self._queue.__getitem__(key)

    def __repr__(self):
        return self._queue.__repr__()

    def _clean(self):
        if len(self._queue) == 0:
            return
        windown_tail = self._queue[-1].time_ns - self._time_window_ns
        while True:
            if self._queue[0].time_ns < windown_tail:
                self._queue.popleft()
            else:
                return
            