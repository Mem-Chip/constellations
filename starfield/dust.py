"""三维尘埃场：平滑的盘状尘埃 × 湍流起伏 + 一批高斯尘埃云。

湍流起伏：观察者周围一个盒子里的对数正态密度场（Kolmogorov 谱、沿银盘压扁），
平均值为 1，所以不改变平均消光，只把尘埃聚成连成片的云和丝、中间留出空隙；太阳附近有个低密度的本地泡。

同一份场既给单颗恒星算消光，也给背景贴图积分（暗带自然与被挡的星一致）。
"""
import numpy as np

from .hashrng import module_rng


def erf(x):
    """Abramowitz-Stegun 7.1.26，误差 < 1.5e-7。"""
    x = np.asarray(x, dtype=np.float64)
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return s * y


class Dust:
    CLOUD_REGION_PC = 3000.0   # 在这个半径内生成离散尘埃云；更远只用平滑场

    def __init__(self, cfg, galaxy):
        d = cfg["dust"]
        self.gal = galaxy
        self.kappa0 = float(d["mean_extinction_mag_per_kpc"]) / 1000.0  # mag/pc（盘面中、本地）
        self.h = float(d["scale_height_pc"])
        self.Rd = float(cfg["galaxy"]["thin_disk"]["scale_length_kpc"]) * 1000.0
        self.Rv = float(d["reddening_Rv"])
        self._make_clouds(cfg)
        self._make_turbulence(cfg)

    def _make_clouds(self, cfg):
        c = self.cfg_clouds = cfg["dust"]["clouds"]
        rng = module_rng(cfg["meta"]["seed"], "dust")
        area_density = float(c["count_within_1kpc"]) / (np.pi * 1000.0 ** 2)
        R = self.CLOUD_REGION_PC
        n = rng.poisson(area_density * np.pi * R ** 2)
        r = R * np.sqrt(rng.random(n))
        th = rng.random(n) * 2 * np.pi
        x, y = r * np.cos(th), r * np.sin(th)
        z_abs = rng.laplace(0.0, self.h, n)
        rad = np.exp(rng.uniform(np.log(c["radius_pc"][0]), np.log(c["radius_pc"][1]), n))
        peak = np.exp(rng.uniform(np.log(c["peak_extinction_mag"][0]), np.log(c["peak_extinction_mag"][1]), n))
        # 每朵云做成丝状：沿一条随机游走的曲线串 SUB 个细团块，总长约 2×半径
        SUB = 12
        dirs = rng.normal(0, 1, (n, 3))
        dirs[:, 2] *= 0.3                                     # 大致顺着银盘
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        cx, cy, cz, cs, cp = [], [], [], [], []
        for k in range(SUB):
            bend = rng.normal(0, 0.22, (n, 3))
            dirs = dirs + bend
            dirs /= np.linalg.norm(dirs, axis=1)[:, None]
            t = (k - (SUB - 1) / 2) / ((SUB - 1) / 2) * rad   # -rad … +rad
            jitter = rng.normal(0, 0.08, n) * rad
            cx.append(x + dirs[:, 0] * (t + jitter))
            cy.append(y + dirs[:, 1] * (t + jitter))
            cz.append(z_abs + dirs[:, 2] * (t + jitter))
            cs.append(rad / 7.0 * np.exp(rng.normal(0, 0.35, n)))
            cp.append(peak * np.exp(rng.normal(0, 0.5, n)))
        x, y, z_abs = np.concatenate(cx), np.concatenate(cy), np.concatenate(cz)
        sigma, peak = np.concatenate(cs), np.concatenate(cp)
        pos = np.column_stack([x, y, z_abs - self.gal.z_obs])
        # 观察者不能被埋在云里
        dist = np.linalg.norm(pos, axis=1)
        keep = dist > 2.5 * sigma + 5.0
        self.c_pos = pos[keep]
        self.c_sigma = sigma[keep]
        self.c_peak = peak[keep]
        self.c_dist = dist[keep]

    # ---------- 湍流起伏 ----------
    def _make_turbulence(self, cfg):
        t = dict(cfg["dust"].get("turbulence") or {})
        self.turb_on = bool(t.get("enabled", True))
        if not self.turb_on:
            return
        v = float(t.get("voxel_pc", 8.0))
        half = float(t.get("box_half_pc", 2400.0))
        zh = float(t.get("z_half_pc", 480.0))
        sig = float(t.get("sigma_ln", 1.3))
        L0 = float(t.get("outer_scale_pc", 800.0))
        flat = float(t.get("flattening", 1.8))
        bub = float(t.get("local_bubble_pc", 80.0))
        nx, nz = int(round(2 * half / v)), int(round(2 * zh / v))
        rng = module_rng(cfg["meta"]["seed"], "dustturb")
        g = rng.standard_normal((nx, nx, nz), dtype=np.float32)
        G = np.fft.rfftn(g)
        del g
        kx = np.fft.fftfreq(nx, v).astype(np.float32)
        kz = np.fft.rfftfreq(nz, v).astype(np.float32)
        k2 = kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + (kz[None, None, :] / flat) ** 2
        G *= ((k2 + (1.0 / L0) ** 2) ** (-11.0 / 12.0)).astype(np.float32)   # 振幅 = sqrt(P)，P ∝ k^-11/3
        del k2
        G[0, 0, 0] = 0
        g = np.fft.irfftn(G, s=(nx, nx, nz)).astype(np.float32)
        del G
        g /= g.std()
        F = np.exp(sig * g)
        del g
        F /= F.mean()
        # 盒子边缘渐变回 1（与外面的平滑场衔接），中心挖出本地泡
        c = (np.arange(nx, dtype=np.float32) + 0.5) * v - half
        cz = (np.arange(nz, dtype=np.float32) + 0.5) * v - zh
        rxy = np.hypot(c[:, None], c[None, :])
        taper = 1.0 - np.clip((rxy - 0.8 * half) / (0.17 * half), 0, 1)
        F = 1.0 + (F - 1.0) * taper[:, :, None]
        z_rel = cz - self.gal.z_obs
        r3 = np.sqrt(rxy[:, :, None] ** 2 + z_rel[None, None, :] ** 2)
        x = np.clip((r3 - 0.5 * bub) / bub, 0, 1)
        F *= 0.12 + 0.88 * x * x * (3 - 2 * x)
        self.turb = (F - 1.0).astype(np.float32)
        self.t_v, self.t_half, self.t_zh = v, half, zh

    def turb_sample(self, x, y, Z):
        """湍流起伏 F−1 在观察者坐标 (x, y) 与盘面高度 Z 处的三线性插值；盒子外为 0。"""
        T = self.turb
        nx, _, nz = T.shape
        fx = (x + self.t_half) / self.t_v - 0.5
        fy = (y + self.t_half) / self.t_v - 0.5
        fz = (Z + self.t_zh) / self.t_v - 0.5
        ok = (fx >= 0) & (fx < nx - 1) & (fy >= 0) & (fy < nx - 1) & (fz >= 0) & (fz < nz - 1)
        out = np.zeros(np.shape(x), np.float32)
        if not ok.any():
            return out
        fx, fy, fz = fx[ok], fy[ok], fz[ok]
        ix, iy, iz = fx.astype(np.int64), fy.astype(np.int64), fz.astype(np.int64)
        ax, ay, az = (fx - ix).astype(np.float32), (fy - iy).astype(np.float32), (fz - iz).astype(np.float32)
        flat = T.ravel()
        sy, sx = nz, nx * nz
        base = ix * sx + iy * sy + iz
        acc = np.zeros(ix.shape, np.float32)
        for dx in (0, 1):
            wx = ax if dx else 1 - ax
            for dy in (0, 1):
                wy = ay if dy else 1 - ay
                for dz in (0, 1):
                    wz = az if dz else 1 - az
                    acc += wx * wy * wz * flat[base + dx * sx + dy * sy + dz]
        out[ok] = acc
        return out

    def turbulence(self, u, d, nstep=48, chunk=80_000):
        """沿视线 ∫ 平滑密度 × (F−1) ds：湍流对消光的修正（可正可负）。"""
        u = np.asarray(u, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        A = np.zeros(d.shape)
        if not self.turb_on:
            return A
        z0 = self.gal.z_obs
        for i0 in range(0, len(d), chunk):
            uu = u[i0:i0 + chunk]
            dd = d[i0:i0 + chunk]
            uz = uu[:, 2]
            with np.errstate(divide="ignore"):
                sz = np.where(uz > 1e-9, (self.t_zh - z0) / uz, np.where(uz < -1e-9, (-self.t_zh - z0) / uz, np.inf))
                sxy = 0.97 * self.t_half / np.maximum(np.abs(uu[:, 0]), np.abs(uu[:, 1]))
            send = np.minimum(dd, np.minimum(sz, sxy))
            jit = (np.sin(uu @ np.array([12.9898, 78.233, 37.719])) * 43758.5453) % 1.0
            sk = send[:, None] * ((np.arange(nstep)[None, :] + jit[:, None]) / nstep)
            X = uu[:, 0:1] * sk
            Y = uu[:, 1:2] * sk
            Z = z0 + uu[:, 2:3] * sk
            rho = self.gal.radial_factor(X, Y, self.Rd) * np.exp(-np.abs(Z) / self.h)
            f = self.turb_sample(X, Y, Z)
            A[i0:i0 + chunk] = self.kappa0 * np.sum(rho * f, axis=1) * send / nstep
        return A

    # ---------- 平滑场 ----------
    def smooth(self, u, d, nseg=16):
        """沿方向 u (N,3) 从 0 积分到 d (N,) 的平滑场消光。z 方向精确积分，径向按段取中点。"""
        u = np.asarray(u)
        d = np.asarray(d, dtype=np.float64)
        h = self.h
        z0 = self.gal.z_obs
        uz = u[:, 2]
        A = np.zeros(d.shape)
        F = lambda z: np.sign(z) * h * (1.0 - np.exp(-np.abs(z) / h))
        for k in range(nseg):
            sa = d * (k / nseg)
            sb = d * ((k + 1) / nseg)
            sm = 0.5 * (sa + sb)
            rf = self.gal.radial_factor(u[:, 0] * sm, u[:, 1] * sm, self.Rd)
            za = z0 + uz * sa
            zb = z0 + uz * sb
            dz = zb - za
            flat = np.abs(dz) < 1e-3 * h
            safe_uz = np.where(flat, 1.0, uz)
            seg = np.where(flat, np.exp(-np.abs(0.5 * (za + zb)) / h) * (sb - sa), (F(zb) - F(za)) / safe_uz)
            A += rf * seg
        return self.kappa0 * A

    # ---------- 尘埃云 ----------
    def clouds(self, u, d, chunk=2_000_000):
        u = np.asarray(u)
        d = np.asarray(d, dtype=np.float64)
        A = np.zeros(d.shape)
        for i0 in range(0, len(d), chunk):
            uu = u[i0:i0 + chunk]
            dd = d[i0:i0 + chunk]
            AA = A[i0:i0 + chunk]
            for c, s, pk, cd in zip(self.c_pos, self.c_sigma, self.c_peak, self.c_dist):
                s0 = uu @ c
                p2 = cd * cd - s0 * s0
                m = np.nonzero((p2 < 16.0 * s * s) & (s0 > -4.0 * s) & (dd > s0 - 4.0 * s))[0]
                if m.size == 0:
                    continue
                k = 1.0 / (np.sqrt(2.0) * s)
                AA[m] += pk * np.exp(-p2[m] / (2 * s * s)) * 0.5 * (erf((dd[m] - s0[m]) * k) + erf(s0[m] * k))
        return A

    def extinction(self, u, d):
        """V 波段总消光（星等）。"""
        return np.maximum(self.smooth(u, d) + self.turbulence(u, d), 0.0) + self.clouds(u, d)
