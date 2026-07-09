import numpy as np
from dataclasses import dataclass, field

# Constants
STATE_SIZE = 6
STATE_COVAR_SIZE = STATE_SIZE * STATE_SIZE

@dataclass
class State:
    # Class constants
    # Data fields
    
    _mean: np.ndarray = field(default_factory=lambda: np.zeros(STATE_SIZE, dtype=np.float64))
    _sqrt_covar: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 0.3, 0.7, 0.7, 0.3]).astype(np.float64))
    
    def deep_copy(self) -> 'State':
        """Return a deep copy of the State instance."""
        return State(
            _mean=self._mean.copy(),
            _sqrt_covar=self._sqrt_covar.copy()
        )
    
    @property
    def x(self) -> float:
        return self._mean[0]
    
    @property
    def y(self) -> float:
        return self._mean[1]
    
    @property
    def psi(self) -> float:
        return self._mean[2]
        
    @psi.setter
    def psi(self, value):
        self._mean[2] = value

    @property
    def u(self) -> float:
        return self._mean[3]
    
    @property
    def v(self) -> float:
        return self._mean[4]
    
    @property
    def omega(self) -> float:
        return self._mean[5]
    
    def __str__(self) -> str:
        """String representation similar to the C++ operator<<"""
        output = "State Mean: \n"
        output += " | ".join(f"{val:.6f}" for val in self._mean) + "\n"
        output += "State Covar: \n"
        
        # Format covariance matrix in a readable way
        for i in range(STATE_SIZE):
            row = self._sqrt_covar[i, :]
            output += " | ".join(f"{val:.6f}" for val in row) + "\n"
        
        return output
    
    def __repr__(self) -> str:
        return f"State(mean={self._mean}, covar_shape={self._sqrt_covar.shape})"
    
    def normalize(self) -> None:
        """Normalize rotations to the range [0, 2pi]"""
        self.psi = np.fmod(self.psi, 2*np.pi)
        if self.psi < 0:
            self.psi += 2*np.pi