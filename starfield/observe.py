"""观测层：从观察者位置推导天球位置、视星等、颜色、视差、自行。"""
import numpy as np

from . import evolution as E
from .coords import apply, lonlat
from .galaxy import KMS_TO_PCYR

K_PM = 4.740470   # km/s 对应 1 AU/yr


def add_extinction(stars, dust):
    d = stars["d"]
    u = np.column_stack([stars["x"], stars["y"], stars["z"]]) / d[:, None]
    stars["A_V"] = dust.extinction(u, d)
    stars["MV_sys"] = stars.get("MV_sys", stars["MV"])
    stars["V"] = stars["MV_sys"] + 5 * np.log10(d) - 5 + stars["A_V"]
    stars["BV0"] = E.bv_from_teff(stars["Teff"])
    stars["BV"] = stars["BV0"] + stars["A_V"] / dust.Rv


RES_NONE, RES_TELESCOPE, RES_NAKED = 0, 1, 2


def binary_resolvability(stars, cfg):
    """每颗伴星：视星等、角距离、能否分辨（0 不能 / 1 望远镜 / 2 肉眼）。"""
    ob = cfg["observing"]
    ne, tel = ob["naked_eye"], ob["telescope"]
    dm_max = float(ob["binary_resolvability"]["max_mag_difference"])
    DM = stars["V"] - stars["MV_sys"]            # 距离模数 + 消光
    Vp = stars["MV"] + DM
    stars["V_primary"] = Vp
    for c in (1, 2):
        pre = f"c{c}_"
        Vc = stars[pre + "MV"] + DM
        sep = stars[pre + "sep_au"] / stars["d"]
        dm = np.abs(Vc - Vp)
        ok_dm = (dm <= dm_max) | (sep > 30 * float(tel["resolution_arcsec"]))   # 相距很远时亮度差不碍事
        naked = (sep >= ne["resolution_arcsec"]) & (Vp <= ne["limit_mag"]) & (Vc <= ne["limit_mag"]) & ok_dm
        teles = (sep >= tel["resolution_arcsec"]) & (Vc <= tel["limit_mag"]) & ok_dm
        res = np.where(naked, RES_NAKED, np.where(teles, RES_TELESCOPE, RES_NONE)).astype(np.int8)
        res[~np.isfinite(sep)] = RES_NONE
        stars[pre + "V"] = Vc
        stars[pre + "sep_arcsec"] = sep
        stars[pre + "resolvable"] = res
    return stars


def subset(stars, mask):
    n = len(mask)
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in stars.items()}


def sky_positions(stars, clock, year):
    """某年代的位置：线性推进三维运动，再转到黄道与赤道坐标。返回新字典（不改原表）。"""
    dt = year - clock.ref_year
    p = np.column_stack([stars["x"], stars["y"], stars["z"]])
    v = np.column_stack([stars["vx"], stars["vy"], stars["vz"]])
    p = p + v * (KMS_TO_PCYR * dt)
    d = np.linalg.norm(p, axis=1)
    u = p / d[:, None]
    out = {"d": d}
    out["gal_lon"], out["gal_lat"] = lonlat(u)
    M_ecl = clock.gal_to_ecl_date(year)
    out["ecl_lon"], out["ecl_lat"] = lonlat(apply(M_ecl, u))
    M_equ = clock.gal_to_equ(year)
    ue = apply(M_equ, u)
    ve = apply(M_equ, v)
    ra, dec = lonlat(ue)
    out["ra"], out["dec"] = ra, dec
    a, dl = np.radians(ra), np.radians(dec)
    e_ra = np.column_stack([-np.sin(a), np.cos(a), np.zeros_like(a)])
    e_dec = np.column_stack([-np.sin(dl) * np.cos(a), -np.sin(dl) * np.sin(a), np.cos(dl)])
    out["rv"] = np.sum(ve * ue, axis=1)
    out["pm_ra"] = np.sum(ve * e_ra, axis=1) / (K_PM * d) * 1000.0      # mas/yr（已含 cosδ）
    out["pm_dec"] = np.sum(ve * e_dec, axis=1) / (K_PM * d) * 1000.0
    out["parallax_mas"] = 1000.0 / d
    return out
