"""Geodesic helpers and coarse Malta/Gozo locality assignment.

Locality centroids are approximate town-centre coordinates intended for
coarse regional aggregation (nearest-centroid assignment), not for exact
administrative boundary work.  Each locality is tagged with its NSO
statistical district so captures can also be summarised per district.
"""

from __future__ import annotations

import math

import pandas as pd

# (locality, district, latitude, longitude) - approximate town centres.
MALTA_LOCALITIES: list[tuple[str, str, float, float]] = [
    ("Valletta", "Southern Harbour", 35.8989, 14.5146),
    ("Floriana", "Southern Harbour", 35.8938, 14.5056),
    ("Marsa", "Southern Harbour", 35.8792, 14.4953),
    ("Paola", "Southern Harbour", 35.8722, 14.4989),
    ("Tarxien", "Southern Harbour", 35.8672, 14.5117),
    ("Fgura", "Southern Harbour", 35.8706, 14.5236),
    ("Zabbar", "Southern Harbour", 35.8764, 14.5350),
    ("Cospicua", "Southern Harbour", 35.8806, 14.5222),
    ("Vittoriosa", "Southern Harbour", 35.8878, 14.5225),
    ("Senglea", "Southern Harbour", 35.8872, 14.5169),
    ("Kalkara", "Southern Harbour", 35.8892, 14.5328),
    ("Xghajra", "Southern Harbour", 35.8853, 14.5475),
    ("Luqa", "Southern Harbour", 35.8592, 14.4886),
    ("Santa Lucija", "Southern Harbour", 35.8608, 14.5042),
    ("Sliema", "Northern Harbour", 35.9122, 14.5042),
    ("Gzira", "Northern Harbour", 35.9058, 14.4953),
    ("Msida", "Northern Harbour", 35.8956, 14.4894),
    ("Ta' Xbiex", "Northern Harbour", 35.8992, 14.4944),
    ("Pieta", "Northern Harbour", 35.8928, 14.4950),
    ("Hamrun", "Northern Harbour", 35.8861, 14.4894),
    ("Santa Venera", "Northern Harbour", 35.8908, 14.4778),
    ("Birkirkara", "Northern Harbour", 35.8972, 14.4611),
    ("San Gwann", "Northern Harbour", 35.9083, 14.4756),
    ("St Julian's", "Northern Harbour", 35.9186, 14.4889),
    ("Swieqi", "Northern Harbour", 35.9228, 14.4800),
    ("Pembroke", "Northern Harbour", 35.9308, 14.4767),
    ("Qormi", "Northern Harbour", 35.8764, 14.4719),
    ("Zebbug (Malta)", "Western", 35.8722, 14.4392),
    ("Siggiewi", "Western", 35.8556, 14.4364),
    ("Dingli", "Western", 35.8608, 14.3814),
    ("Rabat (Malta)", "Western", 35.8817, 14.3989),
    ("Mdina", "Western", 35.8858, 14.4028),
    ("Mtarfa", "Western", 35.8892, 14.3944),
    ("Attard", "Western", 35.8897, 14.4425),
    ("Balzan", "Western", 35.8983, 14.4550),
    ("Lija", "Western", 35.9006, 14.4472),
    ("Iklin", "Western", 35.9075, 14.4522),
    ("Birzebbuga", "South Eastern", 35.8258, 14.5269),
    ("Marsaxlokk", "South Eastern", 35.8419, 14.5442),
    ("Zejtun", "South Eastern", 35.8556, 14.5333),
    ("Marsaskala", "South Eastern", 35.8622, 14.5675),
    ("Ghaxaq", "South Eastern", 35.8489, 14.5169),
    ("Gudja", "South Eastern", 35.8492, 14.5031),
    ("Kirkop", "South Eastern", 35.8422, 14.4853),
    ("Mqabba", "South Eastern", 35.8442, 14.4669),
    ("Qrendi", "South Eastern", 35.8347, 14.4581),
    ("Safi", "South Eastern", 35.8331, 14.4850),
    ("Zurrieq", "South Eastern", 35.8311, 14.4742),
    ("Mosta", "Northern", 35.9092, 14.4256),
    ("Naxxar", "Northern", 35.9136, 14.4436),
    ("Gharghur", "Northern", 35.9225, 14.4517),
    ("Madliena", "Northern", 35.9264, 14.4703),
    ("Bahar ic-Caghaq", "Northern", 35.9331, 14.4533),
    ("Burmarrad", "Northern", 35.9333, 14.4189),
    ("St Paul's Bay", "Northern", 35.9483, 14.4014),
    ("Bugibba", "Northern", 35.9494, 14.4108),
    ("Qawra", "Northern", 35.9525, 14.4194),
    ("Xemxija", "Northern", 35.9494, 14.3861),
    ("Mellieha", "Northern", 35.9564, 14.3622),
    ("Manikata", "Northern", 35.9436, 14.3517),
    ("Mgarr (Malta)", "Northern", 35.9192, 14.3664),
    ("Zebbiegh", "Northern", 35.9231, 14.3781),
    ("Victoria (Gozo)", "Gozo", 36.0444, 14.2397),
    ("Xewkija", "Gozo", 36.0331, 14.2581),
    ("Ghajnsielem", "Gozo", 36.0269, 14.2853),
    ("Mgarr (Gozo)", "Gozo", 36.0253, 14.2950),
    ("Qala", "Gozo", 36.0347, 14.3097),
    ("Nadur", "Gozo", 36.0378, 14.2942),
    ("Xaghra", "Gozo", 36.0500, 14.2644),
    ("Marsalforn", "Gozo", 36.0717, 14.2558),
    ("Zebbug (Gozo)", "Gozo", 36.0656, 14.2358),
    ("Gharb", "Gozo", 36.0603, 14.2089),
    ("San Lawrenz", "Gozo", 36.0553, 14.2036),
    ("Kercem", "Gozo", 36.0400, 14.2269),
    ("Munxar", "Gozo", 36.0308, 14.2331),
    ("Xlendi", "Gozo", 36.0289, 14.2158),
    ("Sannat", "Gozo", 36.0244, 14.2436),
    ("Fontana", "Gozo", 36.0378, 14.2358),
    ("Comino", "Gozo", 36.0119, 14.3369),
]

EARTH_RADIUS_KM = 6371.0088

# Rough bounding box around the Maltese archipelago; GPS fixes outside it are
# treated as receiver glitches and excluded from geographic aggregation.
MALTA_LAT_RANGE = (35.7, 36.2)
MALTA_LON_RANGE = (14.1, 14.7)


def within_malta(points: pd.DataFrame) -> pd.Series:
    """Boolean mask: rows whose ``gps_latitude``/``gps_longitude`` fall in Malta."""
    return (
        points["gps_latitude"].between(*MALTA_LAT_RANGE)
        & points["gps_longitude"].between(*MALTA_LON_RANGE)
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points, in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def locality_frame() -> pd.DataFrame:
    """The locality reference table as a DataFrame."""
    return pd.DataFrame(
        MALTA_LOCALITIES, columns=["locality", "district", "latitude", "longitude"]
    )


def assign_localities(points: pd.DataFrame) -> pd.DataFrame:
    """Assign each GPS point to its nearest locality centroid.

    ``points`` needs ``gps_latitude``/``gps_longitude`` columns; the result
    adds ``locality``, ``district`` and ``locality_distance_km``.
    """
    reference = locality_frame()
    localities, districts, distances = [], [], []
    for lat, lon in zip(points["gps_latitude"], points["gps_longitude"]):
        best_idx, best_km = None, float("inf")
        for idx, ref in enumerate(MALTA_LOCALITIES):
            km = haversine_km(lat, lon, ref[2], ref[3])
            if km < best_km:
                best_idx, best_km = idx, km
        localities.append(reference.at[best_idx, "locality"])
        districts.append(reference.at[best_idx, "district"])
        distances.append(round(best_km, 3))

    result = points.copy()
    result["locality"] = localities
    result["district"] = districts
    result["locality_distance_km"] = distances
    return result
