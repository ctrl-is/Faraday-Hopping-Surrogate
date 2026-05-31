from dataclasses import dataclass
from enum import Enum
import math

class Region(Enum):
    A = "A" # x, y > 0
    B = "B" # x < 0, y > 0
    C = "C" # x, y < 0
    D = "D" # x > 0, y < 0
    GAP = "gap" # within the physical gap of grid quadrants
    OUTSIDE = "outside" # outside grid

@dataclass
class CollectorGeometry:
    gap_width: float
    radius: float

    def is_inside_collector(self, coords: tuple) -> bool:
        x, y = coords
        return x**2 + y**2 <= self.radius**2
    
    def is_inside_gap(self, coords: tuple) -> bool:
        x, y = coords
        return abs(x) < self.gap_width / 2 or abs(y) < self.gap_width / 2
    
    def region_at(self, coords: tuple) -> Region:
        x, y = coords
        if not self.is_inside_collector(coords):
            return Region.OUTSIDE
        
        if self.is_inside_gap(coords):
            return Region.GAP
        
        if x > 0 and y > 0:
            return Region.A
        elif x < 0 and y > 0:
            return Region.B
        elif x < 0 and y < 0:
            return Region.C
        elif x > 0 and y < 0:
            return Region.D
        
        return Region.GAP # case where x, y = 0, 0

