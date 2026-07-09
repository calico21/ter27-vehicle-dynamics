import math
from typing import Optional, NamedTuple

class Point2D(NamedTuple):
    """Represents a 2D point with x and y coordinates"""
    x: float
    y: float


class GpsReceiver:
    """
    Receives GPS messages and translates them to local coordinate frame
    """
    
    def __init__(self):
        self._initialized = False
        self._lat_ref = 0.0  # Reference latitude in degrees
        self._lon_ref = 0.0  # Reference longitude in degrees
        self._x_ref = 0.0    # Reference point in meters
        self._y_ref = 0.0    # Reference point in meters
        self._lat_meters_per_deg = 0.0  # Latitude degrees to meters near the reference frame
        self._lon_meters_per_deg = 0.0  # Longitude degrees to meters near the reference frame

    def receive_gps(self, lat: float, lon: float) -> Optional[Point2D]:
        """
        Returns the point in 2D that corresponds to the given lat & longitude 
        relative to the values with which the receiver was initialized.
        
        Returns None if the receiver has not been initialized.
        """
        if not self._initialized:
            return None
            
        x = self._x_ref + (lat - self._lat_ref) * self._lat_meters_per_deg
        y = self._y_ref + (lon - self._lon_ref) * self._lon_meters_per_deg
        
        return Point2D(x, y)

    def initialize(self, lat_ref: float, lon_ref: float, x_ref: float, y_ref: float) -> None:
        """
        Establish a reference frame by giving a (latitude, longitude) pair and 
        the (x, y) position in meters it corresponds to.
        """
        # Set ref point that corresponds to (0, 0)
        self._lat_ref = lat_ref
        self._lon_ref = lon_ref
        self._x_ref = x_ref
        self._y_ref = y_ref

        # Obtain the conversion rate from degrees to meters assuming we are close to that reference point
        # Using WGS84 ellipsoid parameters
        lat_ref_rad = math.radians(lat_ref)
        
        # Calculate meters per degree for latitude
        self._lat_meters_per_deg = (
            111132.92 
            - 559.82 * math.cos(2 * lat_ref_rad) 
            + 1.175 * math.cos(4 * lat_ref_rad)
        )
        
        # Calculate meters per degree for longitude
        self._lon_meters_per_deg = 111320.0 * math.cos(lat_ref_rad)

        self._initialized = True

    def is_initialized(self) -> bool:
        """Check if the GPS receiver has been initialized"""
        return self._initialized