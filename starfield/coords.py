"""坐标系链：银道（固定）→ 黄道（固定）→ 某年代赤道 → 地平。

银道直角坐标约定：x 指向银心（l=0），y 指向 l=90°，z 指向银北极；原点在观察者。
"""
import numpy as np

D2R = np.pi / 180.0


def unit(lon_deg, lat_deg):
    lon = np.asarray(lon_deg, dtype=np.float64) * D2R
    lat = np.asarray(lat_deg, dtype=np.float64) * D2R
    c = np.cos(lat)
    return np.stack([c * np.cos(lon), c * np.sin(lon), np.sin(lat)], axis=-1)


def lonlat(v):
    v = np.asarray(v, dtype=np.float64)
    r = np.linalg.norm(v, axis=-1)
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360.0
    lat = np.degrees(np.arcsin(np.clip(v[..., 2] / np.maximum(r, 1e-300), -1, 1)))
    return lon, lat


def rot_x(a_deg):
    a = a_deg * D2R
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a_deg):
    a = a_deg * D2R
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def gal_to_ecl_matrix(cfg):
    """v_ecl = M @ v_gal。

    银道面在黄道上的升交点黄经 Ω，倾角 i；银心位于沿银道从升交点量起 u0 处。
    """
    o = cfg["galaxy"]["orientation"]
    om = o["galactic_ascending_node_ecl_lon_deg"] * D2R
    inc = o["ecliptic_galactic_inclination_deg"] * D2R
    u0 = o["galactic_center_from_node_deg"] * D2R
    n = np.array([np.cos(om), np.sin(om), 0.0])
    m = np.array([-np.sin(om) * np.cos(inc), np.cos(om) * np.cos(inc), np.sin(inc)])
    ex = n * np.cos(u0) + m * np.sin(u0)       # 银心方向
    ey = -n * np.sin(u0) + m * np.cos(u0)      # l = 90°
    ez = np.cross(ex, ey)                       # 银北极
    return np.stack([ex, ey, ez], axis=1)


class SkyClock:
    """岁差与黄赤夹角随年代的变化。year 用故事纪年。"""

    def __init__(self, cfg):
        sc = cfg["sky_clock"]
        self.P = float(sc["precession"]["period_yr"])
        rot = sc["rotation"]
        self.eps0 = float(rot["obliquity_deg"])
        osc = rot.get("obliquity_oscillation", {}) or {}
        self.eps_amp = float(osc.get("amplitude_deg", 0)) if osc.get("enabled") else 0.0
        self.eps_per = float(osc.get("period_yr", 41000))
        self.lam0 = float(rot.get("equinox_ecl_lon_at_reference_deg", 0))
        self.ref_year = float(cfg["output"]["epochs"]["present_year"])
        self.M_g2e = gal_to_ecl_matrix(cfg)

    def precession_deg(self, year):
        return 360.0 * (year - self.ref_year) / self.P

    def obliquity_deg(self, year):
        return self.eps0 + self.eps_amp * np.sin(2 * np.pi * (year - self.ref_year) / self.eps_per)

    def ecl_to_ecl_of_date(self, year):
        """固定黄道 → 当年春分点起算的黄道（星的黄经随岁差增加）。"""
        return rot_z(self.precession_deg(year) - self.lam0)

    def ecl_of_date_to_equ(self, year):
        return rot_x(self.obliquity_deg(year))

    def gal_to_equ(self, year):
        return self.ecl_of_date_to_equ(year) @ self.ecl_to_ecl_of_date(year) @ self.M_g2e

    def gal_to_ecl_date(self, year):
        return self.ecl_to_ecl_of_date(year) @ self.M_g2e


def apply(M, v):
    """对 (..., 3) 向量组应用 3x3 矩阵。"""
    return np.asarray(v) @ np.asarray(M).T


def equ_to_horizon(ra_deg, dec_deg, lst_deg, lat_deg):
    """赤道 → 地平（方位从北向东量）。"""
    H = (lst_deg - ra_deg) * D2R
    d = np.asarray(dec_deg) * D2R
    phi = lat_deg * D2R
    alt = np.arcsin(np.sin(d) * np.sin(phi) + np.cos(d) * np.cos(phi) * np.cos(H))
    az = np.arctan2(-np.cos(d) * np.sin(H), np.sin(d) * np.cos(phi) - np.cos(d) * np.sin(phi) * np.cos(H))
    return np.degrees(az) % 360.0, np.degrees(alt)
