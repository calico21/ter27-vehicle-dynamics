from collections import deque
from dataclasses import dataclass
from typing import Literal, Union
import numpy as np
import math

from .filters import Filters
from .state import State
from .control_queue import ControlQueue
from .time_windowed_queue import TimeWindowedQueue

class Measure():
    Type = Literal["startup", "imu", "gps", "gss"]

    def __init__(self, measure_: np.array, measure_type_: Type):
        self.measure = measure_
        self.type = measure_type_

    def __repr__(self):
        return f"Measure: {self.measure}, Type: {self.type}"


class Estimator:

    @dataclass
    class QueueEntry():
        measure: Measure
        cached_state_post_update: State


    def __init__(self, filters_: Filters, initial_time_ns: int, time_windown_ns: int, max_ns_between_updates: int = 1e7):
        self._tail_state = State()        
        self._tail_time_ns = initial_time_ns
        self._head_state = None
        self._head_time_ns = initial_time_ns
        self._cached_states_are_valid = False

        self._measure_queue: TimeWindowedQueue[Estimator.QueueEntry] = TimeWindowedQueue(time_windown_ns)
        self._measure_queue.insert(self._tail_time_ns, Estimator.QueueEntry(Measure(None, "startup"), self._tail_state))
        self._control_queue = ControlQueue(math.ceil(time_windown_ns*2))
        
        self._time_window_ns = time_windown_ns
        self._max_h_ns = max_ns_between_updates
        self._filters = filters_

        # Covariance matrices
        self._sqrt_Q = np.diag([0.1, 0.1, 0.03, 0.3, 0.3, 0.03]).astype(np.float64)

        self._sqrt_R_gps = np.diag([8, 8]).astype(np.float64)
        self._sqrt_R_imu = np.diag([0.1, 0.03]).astype(np.float64)
        self._sqrt_R_gss = np.diag([0.3, 0.22]).astype(np.float64)


    # Returns true if everything went ok
    def insert_measure(self, time_ns: int, measure: Measure) -> None:
        # Insert keeping measure order
        
        self._measure_queue.insert(time_ns, Estimator.QueueEntry(measure, None))
        self._invalidate_caches_later_than(time_ns, True)
        if not self._validate_cached_states():
            return
        self._update_tail();        

    def get_estimate(self, time_ns: int) -> Union[State, None]:
        if time_ns < self._head_time_ns:
            self._head_state = None

        if self._head_state is not None and self._head_time_ns < time_ns:
            last_time_ns = self._head_time_ns
            last_state = self._head_state
        else:
            last_measurement = self._measure_queue[-1]
            last_time_ns = last_measurement.time_ns
            if not self._validate_cached_states() or time_ns < last_time_ns:
                return None
            last_state = last_measurement.content.cached_state_post_update
            
        predicted_state = self._predict_or_update(time_ns, last_time_ns, last_state, None)
        if predicted_state is None: 
            return None
        self._head_time_ns = time_ns
        self._head_state = predicted_state
        return predicted_state.deep_copy()

    def _predict_or_update(self, target_time_ns: int, from_time_ns: int, from_state: State, measure: Union[Measure, None]) -> Union[State, None]:
        """If measure is not None, performs update. Else, perfroms predict"""
        time_diff_ns: int = target_time_ns - from_time_ns
        if (time_diff_ns < 0): return from_state
        steps = math.ceil(time_diff_ns / self._max_h_ns)
        if steps <= 0:
            return from_state.deep_copy()
        h_ns = time_diff_ns / steps


        # Ok lemme comment this real quick cause this is a fucking huge mess.
        # To predict from t0 to t1, we perform N prediction steps with dt = t0-t1/h
        # If we have measure, the last step is a measured update.
        curr_state_estimate = from_state.deep_copy()

        # N predict steps (or n-1 if there is a measure at the end)
        for step in range(steps - int(measure is not None)):
            # The prediction function behaves equaly for all 3 filters so it doesnt matter
            next_x, next_sqrt_P = self._filters.gps_kalman.predict(
                curr_state_estimate._mean, 
                curr_state_estimate._sqrt_covar, 
                np.concatenate([self._get_control_at_or_default(from_time_ns+ step*h_ns), [h_ns/1e9]]), # Control = [..., timestep] 
                self._sqrt_Q
                )
            curr_state_estimate._mean = np.array(next_x).flatten()
            curr_state_estimate._sqrt_covar = np.array(next_sqrt_P)
        
        # The last step is an update step
        if measure is not None:
            step = steps-1
            match measure.type:
                case "startup":
                    next_x, next_sqrt_P = self._filters.gps_kalman.predict(
                        curr_state_estimate._mean, 
                        curr_state_estimate._sqrt_covar, 
                        np.concatenate([self._get_control_at_or_default(from_time_ns+ step*h_ns), [h_ns/1e9]]), # Control = [..., timestep] 
                        self._sqrt_Q
                        )
                    curr_state_estimate._mean = np.array(next_x).flatten()
                    curr_state_estimate._sqrt_covar = np.array(next_sqrt_P)
                case "gps":
                    next_x, next_sqrt_P = self._filters.gps_kalman.update(
                        curr_state_estimate._mean,
                        curr_state_estimate._sqrt_covar,
                np.concatenate([self._get_control_at_or_default(from_time_ns + step*h_ns), [h_ns/1e9]]), # Control = [..., timestep] 
                        measure.measure,
                        self._sqrt_Q,
                        self._sqrt_R_gps
                        )
                    curr_state_estimate._mean = np.array(next_x).flatten()
                    curr_state_estimate._sqrt_covar = np.array(next_sqrt_P)
                case "imu":

                    # Normalize yaw and make yaw continuous around yaw = 0
                    yaw = measure.measure[0]
                    if(yaw < 0):
                        yaw += 2*np.pi
        
                    diff = yaw - curr_state_estimate.psi
                    if(diff > np.pi):
                        # If there is a positive jump from state.psi to yaw 
                        yaw -= 2*np.pi
                    elif(diff < -np.pi):
                        # If there is a negative jump from state.psi to yaw 
                        yaw += 2*np.pi

                    _normalized_measure = np.array([yaw, measure.measure[1]], dtype = np.float64)

                    next_x, next_sqrt_P = self._filters.imu_kalman.update(
                        curr_state_estimate._mean,
                        curr_state_estimate._sqrt_covar,
                        np.concatenate([self._get_control_at_or_default(from_time_ns + step*h_ns), [h_ns/1e9]]), # Control = [..., timestep] 
                        _normalized_measure,
                        self._sqrt_Q,
                        self._sqrt_R_imu
                        )
                    curr_state_estimate._mean = np.array(next_x).flatten()
                    curr_state_estimate._sqrt_covar = np.array(next_sqrt_P)
                case "gss":
                    next_x, next_sqrt_P = self._filters.gss_kalman.update(
                        curr_state_estimate._mean,
                        curr_state_estimate._sqrt_covar,
                        np.concatenate([self._get_control_at_or_default(from_time_ns + step*h_ns), [h_ns/1e9]]), # Control = [..., timestep] 
                        measure.measure,
                        self._sqrt_Q,
                        self._sqrt_R_gss
                        )
                    curr_state_estimate._mean = np.array(next_x).flatten()
                    curr_state_estimate._sqrt_covar = np.array(next_sqrt_P)
                case _:
                    raise ValueError(f"Unknown/Invaeeeeeeesure type: {measure.type}")

        return curr_state_estimate

    def _invalidate_caches_later_than(self, time_ns: int, later_or_equal: bool = False):
        index = self._measure_queue.get_earliest_index_later_than(time_ns, later_or_equal)
        if index == len(self._measure_queue):
            return
        for i in range(index, len(self._measure_queue)):
            self._measure_queue[i].content.cached_state_post_update = None
        self._cached_states_are_valid = False
        self._head_state = None

    def _validate_cached_states(self) -> bool:
        """Returns True if all cached states are valid. If they arent, tries to fix them and returns true in case of success"""
        if self._cached_states_are_valid:
            return True 
        
        last_valid_cache_index = -1
        for i in range(len(self._measure_queue)):
            if self._measure_queue[i].content.cached_state_post_update is not None:
                last_valid_cache_index = i
            else:
                break

        while last_valid_cache_index < len(self._measure_queue) - 1:
            if last_valid_cache_index < 0:
                last_time_ns = self._tail_time_ns
                last_state = self._tail_state
            else:
                entry = self._measure_queue[last_valid_cache_index]
                last_time_ns = entry.time_ns
                last_state = entry.content.cached_state_post_update
            next_entry = self._measure_queue[last_valid_cache_index+1]
            result = self._predict_or_update(next_entry.time_ns, last_time_ns, last_state, next_entry.content.measure)
            if result is None:
                return False
            next_entry.content.cached_state_post_update = result
            last_valid_cache_index += 1         
    
        self._cached_states_are_valid = True
        return True


    def _get_control_at_or_default(self, time_ns: int) -> np.array:
        ret = self._control_queue.get_control_at(time_ns)
        return ret if ret is not None else np.array([0, 0], dtype = np.float64)
    
    def insert_control(self, time_ns: int, control: np.array) -> None:
        """Do not include timestep, as last element of the control vector. If the control vector is [g, delta, timestep] insert just [g, delta]"""
        self._control_queue.insert_control(time_ns, control)
        self._invalidate_caches_later_than(time_ns)

    def _update_tail(self):
        target_tail_time = self._measure_queue[-1].time_ns -self._time_window_ns
        if self._tail_time_ns + self._max_h_ns < target_tail_time:
            last_time_ns = self._tail_time_ns
            last_state = self._tail_state
            predicted_state = self._predict_or_update(target_tail_time, last_time_ns, last_state, None)
            if predicted_state is None: 
                return None
            self._tail_time_ns = target_tail_time
            self._tail_state = predicted_state
        # If tail_time < measure_queue.last.time - window_time: tail = predict(at_tail_time+h, tail_time, tail_state) 