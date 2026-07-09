from abc import ABC, abstractmethod
from mirena_common.msg import Track , EntityList
from geometry_msgs.msg import Vector3
from dataclasses import dataclass

@dataclass
class Plan:
    track: Track
    start_point: Vector3 | None
    end_point: Vector3 | None

class BasePlanner(ABC):
    @abstractmethod
    def plan(self, entity_list : EntityList) -> Plan | None :
        pass