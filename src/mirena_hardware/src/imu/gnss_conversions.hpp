#include <cmath>
#include <iostream>

namespace mirena::GNSSConversion {
// Earth's ellipsoid parameters (WGS84)
const double A = 6378137.0;               // Semi-major axis (meters)
const double E2 = 0.00669437999014;       // First eccentricity squared

// Convert degrees to radians
inline double degToRad(double degrees) {
    return degrees * M_PI / 180.0;
}

// Convert radians to degrees
inline double radToDeg(double radians) {
    return radians * 180.0 / M_PI;
}

// Convert geodetic to ECEF coordinates
void geodeticToECEF(double lat_rad, double lon_rad, double alt_m,
                    double& x, double& y, double& z) {
    double sin_lat = sin(lat_rad);
    double cos_lat = cos(lat_rad);
    double sin_lon = sin(lon_rad);
    double cos_lon = cos(lon_rad);
    
    double N = A / sqrt(1.0 - E2 * sin_lat * sin_lat);
    
    x = (N + alt_m) * cos_lat * cos_lon;
    y = (N + alt_m) * cos_lat * sin_lon;
    z = (N * (1.0 - E2) + alt_m) * sin_lat;
}

// Convert latitude/longitude/altitude to NED coordinates
void latLonAltToNED(double lat_deg, double lon_deg, double alt_m,
                    double ref_lat_deg, double ref_lon_deg, double ref_alt_m,
                    double& north, double& east, double& down) {
    // Convert to radians
    double lat = degToRad(lat_deg);
    double lon = degToRad(lon_deg);
    double ref_lat = degToRad(ref_lat_deg);
    double ref_lon = degToRad(ref_lon_deg);
    
    // Convert both points to ECEF
    double x, y, z;
    double ref_x, ref_y, ref_z;
    
    geodeticToECEF(lat, lon, alt_m, x, y, z);
    geodeticToECEF(ref_lat, ref_lon, ref_alt_m, ref_x, ref_y, ref_z);
    
    // Calculate difference
    double dx = x - ref_x;
    double dy = y - ref_y;
    double dz = z - ref_z;
    
    // Rotation matrix from ECEF to NED
    double sin_lat = sin(ref_lat);
    double cos_lat = cos(ref_lat);
    double sin_lon = sin(ref_lon);
    double cos_lon = cos(ref_lon);
    
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz;
    east  = -sin_lon * dx + cos_lon * dy;
    down  = -cos_lat * cos_lon * dx - cos_lat * sin_lon * dy - sin_lat * dz;
}

// Overload with altitude optional (default 0)
void latLonAltToNED(double lat_deg, double lon_deg, double alt_m,
                    double ref_lat_deg, double ref_lon_deg,
                    double& north, double& east, double& down) {
    latLonAltToNED(lat_deg, lon_deg, alt_m, ref_lat_deg, ref_lon_deg, 0.0, north, east, down);
}

// Convert NED back to geodetic (optional utility)
void NEDToLatLonAlt(double north, double east, double down,
                    double ref_lat_deg, double ref_lon_deg, double ref_alt_m,
                    double& lat_deg, double& lon_deg, double& alt_m) {
    double ref_lat = degToRad(ref_lat_deg);
    double ref_lon = degToRad(ref_lon_deg);
    
    double sin_lat = sin(ref_lat);
    double cos_lat = cos(ref_lat);
    double sin_lon = sin(ref_lon);
    double cos_lon = cos(ref_lon);
    
    // Convert NED to ECEF delta
    double dx = -sin_lat * cos_lon * north - sin_lon * east - cos_lat * cos_lon * down;
    double dy = -sin_lat * sin_lon * north + cos_lon * east - cos_lat * sin_lon * down;
    double dz = cos_lat * north - sin_lat * down;
    
    // Get reference ECEF
    double ref_x, ref_y, ref_z;
    geodeticToECEF(ref_lat, ref_lon, ref_alt_m, ref_x, ref_y, ref_z);
    
    // Target ECEF
    double x = ref_x + dx;
    double y = ref_y + dy;
    double z = ref_z + dz;
    
    // Convert ECEF to geodetic (simplified iterative method)
    double p = sqrt(x*x + y*y);
    double lat = atan2(z, p * (1.0 - E2));
    
    // Iterate to refine latitude
    for (int i = 0; i < 5; i++) {
        double N = A / sqrt(1.0 - E2 * sin(lat) * sin(lat));
        double h = p / cos(lat) - N;
        lat = atan2(z, p * (1.0 - E2 * N / (N + h)));
    }
    
    double N = A / sqrt(1.0 - E2 * sin(lat) * sin(lat));
    double h = p / cos(lat) - N;
    double lon = atan2(y, x);
    
    lat_deg = radToDeg(lat);
    lon_deg = radToDeg(lon);
    alt_m = h;
}
}