"""星系结构：盘的密度形状、旋臂、自转曲线与恒星速度。

位置都用"以观察者为原点的银道直角坐标"(pc)；z 方向另存相对银盘中面的绝对高度。
银心系：X = x − R0，Y = y；观察者在 (−R0, 0)，沿 +y 方向公转。
"""
import numpy as np

from .coords import unit

V_CIRC = 220.0          # 平坦自转曲线 (km/s)
KMS_TO_PCYR = 1.0227e-6  # 1 km/s = 1.0227e-6 pc/yr


class Galaxy:
    def __init__(self, cfg):
        g = cfg["galaxy"]
        self.cfg = g
        ob = g["observer"]
        self.R0 = float(ob["galactocentric_kpc"]) * 1000.0
        self.z_obs = float(ob["height_pc"])
        self.v_pec = float(ob["peculiar_velocity_kms"]) * unit(*ob["apex_gal_lon_lat_deg"])
        self.v_obs = np.array([0.0, V_CIRC, 0.0]) + self.v_pec
        self.thin = g["thin_disk"]
        self.thick = g["thick_disk"]

        arms = g.get("spiral_arms", {}) or {}
        self.arms_on = bool(arms.get("enabled")) and g.get("type", "spiral") == "spiral"
        if self.arms_on:
            self.n_arm = int(arms["count"])
            self.pitch = np.radians(float(arms["pitch_angle_deg"]))
            self.arm_sigma = float(arms["width_pc"]) / 2.0
            self.arm_contrast = float(arms["young_star_contrast"])
            off = float(arms["observer_offset_from_arm_pc"])
            # 观察者的银心方位角为 π；让最近的一条臂在垂直于臂的方向上偏开 off
            dphi = off / (self.R0 * np.sin(self.pitch))
            self.phi0 = np.pi - dphi
            spacing = 2 * np.pi * self.R0 * np.tan(self.pitch) / self.n_arm * np.cos(self.pitch)
            self.arm_mean_g = min(1.0, self.arm_sigma * np.sqrt(2 * np.pi) / spacing)

    # ---------- 几何 ----------
    def gc_polar(self, x, y):
        X = np.asarray(x) - self.R0
        Y = np.asarray(y)
        return np.hypot(X, Y), np.arctan2(Y, X)

    def radial_factor(self, x, y, Rd_pc):
        R, _ = self.gc_polar(x, y)
        return np.exp(-(R - self.R0) / Rd_pc)

    def arm_distance(self, x, y):
        """到最近一条旋臂中线的垂直距离 (pc)。"""
        R, phi = self.gc_polar(x, y)
        R = np.maximum(R, 100.0)
        phi_arm = self.phi0 + np.log(R / self.R0) / np.tan(self.pitch)
        period = 2 * np.pi / self.n_arm
        d = (phi - phi_arm + period / 2) % period - period / 2
        return np.abs(d) * R * np.sin(self.pitch)

    def arm_factor(self, x, y, contrast):
        """密度乘数；平均约为 1（对整个盘积分时星数不变）。"""
        c = np.asarray(contrast, dtype=np.float64)
        if not self.arms_on or np.all(c <= 1.001):
            return np.ones(np.shape(x))
        g = np.exp(-0.5 * (self.arm_distance(x, y) / self.arm_sigma) ** 2)
        return (1.0 + (c - 1.0) * g) / (1.0 + (c - 1.0) * self.arm_mean_g)

    def arm_contrast_for_age(self, age_gyr):
        if not self.arms_on:
            return 1.0
        return 1.0 + (self.arm_contrast - 1.0) * np.exp(-age_gyr / 0.15)

    # ---------- 运动 ----------
    def velocities(self, x, y, z_rel, sigma, n1, n2, n3):
        """恒星相对观察者的速度 (km/s)，银道直角坐标。

        sigma: 每颗星的径向速度弥散；n1..n3 为标准正态数。
        """
        X = np.asarray(x) - self.R0
        Y = np.asarray(y)
        R = np.maximum(np.hypot(X, Y), 1.0)
        eR = np.stack([X / R, Y / R], axis=-1)
        ephi = np.stack([Y / R, -X / R], axis=-1)
        sR = sigma
        sphi = 0.65 * sigma
        sz = 0.5 * sigma
        vphi = V_CIRC - sR ** 2 / 80.0 + sphi * n2      # 非对称漂移
        vR = sR * n1
        vxy = eR * vR[:, None] + ephi * vphi[:, None]
        v = np.column_stack([vxy, sz * n3])
        return v - self.v_obs
