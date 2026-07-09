from dataclasses import dataclass
import numpy as np
from typing import Union

from .time_windowed_queue import TimeWindowedQueue

class ControlQueue:

    def __init__(self, time_window_ns: int):
        self._queue = TimeWindowedQueue(time_window_ns)

    def insert_control(self, time_ns: int , control: np.array) -> None:
        self._queue.insert(time_ns, control)

    def get_control_at(self, requested_time_ns: int) -> Union[np.array, None]:
        """Return None if couldnt interpolate"""
        if len(self._queue) == 0:
            return None
        index = self._queue.get_earliest_index_later_than(requested_time_ns)

        if index == 0:
            entry = self._queue[index]
            return entry.content
        elif index == len(self._queue):
            entry = self._queue[index-1]
            return entry.content
        else:
            # interpolate babyyy
            entry_0 = self._queue[index-1]
            entry_1 = self._queue[index]

            weight = (requested_time_ns - entry_0.time_ns) / (entry_1.time_ns - entry_0.time_ns)
            return (1 - weight) * entry_0.content + weight * entry_1.content
        
    def __len__(self):
        return self._queue.__len__()