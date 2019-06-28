import math

R = 6378.137 #[km]
P = 6356.752314 #[km]

def dist(l1, l2):
    lat1, lon1 = l1
    lat2, lon2 = l2

    lat1, lon1 = deg2km(lat1, lon1)
    lat2, lon2 = deg2km(lat2, lon2)

    lat = abs(lat1 - lat2)
    lon = abs(lon1 - lon2)

    return math.sqrt(lat * lat + lon * lon)

def deg2km(lat_deg, lon_deg):

    # latitude
    kmParLat = (2 * math.pi * P) / 360
    lat_km = lat_deg * kmParLat

    # longitude
    r = R * math.cos(math.radians(lat_deg))
    kmParLon = (2 * math.pi * r) / 360
    lon_km = lon_deg * kmParLon

    return (lat_km, lon_km)

def km2deg(lat_km, lon_km):

    # latitude
    latParKm = 360 / (2 * math.pi * P)
    lat_deg = lat_km * latParKm

    # longitude
    r = R * math.cos(math.radians(lat_deg))
    lonParKm = 360 / (2 * math.pi * r)
    lon_deg = lon_km * lonParKm

    return (lat_deg, lon_deg)