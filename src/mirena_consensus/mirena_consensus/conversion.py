from .state import State
from mirena_common.msg import Car

class Conversion:
    def __init__(self):
        raise PermissionError("Conversion is static class")
    
    @staticmethod
    def state_to_msg(state: State) -> Car:
        msg = Car()
        msg.x = state.x
        msg.y = state.y
        msg.psi = state.psi
        msg.u = state.u
        msg.v = state.v
        msg.omega = state.x

        return msg
