"""星空生成入口。

用法：
  .venv/Scripts/python.exe generate.py --mag-limit 6.5      # 小规模验证
  .venv/Scripts/python.exe generate.py                      # 按配置的生成深度（约 11 等）
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from starfield import evolution as E
from starfield.config import load_config
from starfield.coords import SkyClock
from starfield.dust import Dust
from starfield.galaxy import Galaxy
from starfield.background import render_background, save_background
from starfield.clusters import DSO_KINDS, add_remnants, finish_dso, make_groups
from starfield.events import REMNANT_MAG_LIMIT, add_star_supernovae, make_events, make_remnants
from starfield.export import write_all
from starfield.landmarks import apply_landmarks, concat
from starfield.overrides import apply_overrides
from starfield.multiplicity import add_companions
from starfield.observe import add_extinction, binary_resolvability, sky_positions, subset
from starfield.population import IMF, generate_stars
from starfield.variables import VAR_TYPES, assign_eclipsing, assign_pulsators

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass


def log(msg):
    print(msg, flush=True)


REAL_COUNTS = {1.0: 15, 2.0: 48, 3.0: 171, 4.0: 513, 5.0: 1602, 6.0: 4800, 6.5: 9000, 9.0: 120000, 10.0: 350000, 11.0: 1000000}


def summary(stars, mag_lim):
    V = stars["V"]
    log("\n== 累计星数（视星等 ≤ m）  生成 / 真实天空参考 ==")
    for m in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 8.5, 9.0, 10.0, 11.0]:
        if m > mag_lim + 1e-9:
            break
        ref = REAL_COUNTS.get(m)
        log(f"  ≤{m:4.1f}: {int((V <= m).sum()):>9,}" + (f"   / {ref:,}" if ref else ""))

    def composition(mask, title):
        log(f"\n== {title}：光谱型与光度级（共 {int(mask.sum()):,} 颗） ==")
        code, lc = E.classify(stars["Teff"][mask], stars["phase"][mask], stars["L"][mask], stars["mass"][mask])
        letter = np.minimum(code // 10, 6).astype(int)
        dwarf = lc <= 1
        for i, ch in enumerate("OBAFGKM"):
            k = letter == i
            nd, ng = int((k & dwarf).sum()), int((k & ~dwarf & (lc != 6)).sum())
            log(f"  {ch}: 矮星/亚巨星 {nd:>7,}   巨星/超巨星 {ng:>7,}")
        log(f"  白矮星: {int((lc == 6).sum()):,}")
        ph = stars["phase"][mask]
        log("  演化阶段：" + "，".join(f"{E.PHASE_NAMES[p]} {int((ph == p).sum()):,}" for p in range(6)))

    composition(V <= 6.5, "肉眼可见（≤6.5 等）")

    vis = V <= 6.5
    b = stars["gal_lat"][vis]
    log("\n== 肉眼星的银纬分布 ==")
    for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 90)]:
        frac = np.mean((np.abs(b) >= lo) & (np.abs(b) < hi))
        sky = np.sin(np.radians(hi)) - np.sin(np.radians(lo))
        log(f"  |b| {lo:>2}–{hi:<2}°：{frac * 100:5.1f}%   （天区面积占 {sky * 100:4.1f}%）")
    log(f"  距离中位数 {np.median(stars['d'][vis]):.0f} pc；消光中位数 {np.median(stars['A_V'][vis]):.2f} 等")

    log("\n== 最亮的 25 颗 ==")
    order = np.argsort(V)[:25]
    spt = E.spectral_string(*E.classify(stars["Teff"][order], stars["phase"][order], stars["L"][order], stars["mass"][order]))
    for j, i in enumerate(order):
        log(f"  {V[i]:6.2f} 等  M_V {stars['MV'][i]:6.2f}  {spt[j]:<10} {stars['d'][i]:7.1f} pc  "
            f"A_V {stars['A_V'][i]:4.2f}  {stars['mass'][i]:5.2f} M☉  {E.PHASE_NAMES[stars['phase'][i]]:<4} "
            f"l {stars['gal_lon'][i]:5.1f} b {stars['gal_lat'][i]:+5.1f}")


def summary_extra(stars, dso, cfg):
    V = stars["V"]
    ne_lim = cfg["observing"]["naked_eye"]["limit_mag"]
    tel_lim = cfg["observing"]["telescope"]["limit_mag"]
    for lim, title in [(ne_lim, "肉眼"), (tel_lim, "望远镜")]:
        k = V <= lim
        mult = stars["multiplicity"][k]
        r1 = stars["c1_resolvable"][k]
        log(f"\n== {title}可见（≤{lim} 等，{int(k.sum()):,} 颗）的双星与变星 ==")
        log(f"  有伴星 {int((mult >= 2).sum()):,}（三合 {int((mult >= 3).sum()):,}）；"
            f"肉眼可分辨 {int((r1 == 2).sum())}，望远镜可分辨 {int((r1 == 1).sum())}")
        vt = stars["var_type"][k]
        log("  变星：" + "，".join(f"{VAR_TYPES[i]} {int((vt == i).sum())}" for i in range(1, len(VAR_TYPES)) if (vt == i).any()))
        big = k & (stars["var_amp_mag"] >= 0.3)
        log(f"  变幅 ≥0.3 等（肉眼能察觉）的 {int(big.sum())} 颗")
        log(f"  星团成员 {int((stars['cluster_id'][k] >= 0).sum()):,}")
    log("\n== 深空天体（积分星等 ≤ 6，肉眼可见的模糊天体） ==")
    vis = np.nonzero(dso["V_int"] <= 6.0)[0]
    for i in vis[np.argsort(dso["V_int"][vis])][:20]:
        log(f"  {dso['name'][i]:<10} {DSO_KINDS[dso['kind'][i]]:<9} V {dso['V_int'][i]:5.2f}  {dso['d'][i]:8.0f} pc  "
            f"角大小 {dso['ang_size_deg'][i]:5.2f}°  年龄 {dso['age_myr'][i]:7.1f} Myr  l {dso['gal_lon'][i]:5.1f} b {dso['gal_lat'][i]:+5.1f}")


def render_and_save_background(cfg, galaxy, dust, dso, out, bg_compute):
    log("[背景] 全天背景贴图 ...")
    t0 = time.time()
    W, Hh = bg_compute
    bg = render_background(cfg, galaxy, dust, dso, W=W, Hh=Hh, log=log)
    res = cfg["output"]["viewer_data"]["background_map"]["resolution"]
    white = save_background(bg, out / "viewer" / "background.png", out / "background_linear.npy", out_size=res)
    meta_fn = out / "viewer" / "sky_meta.json"
    meta = json.loads(meta_fn.read_text(encoding="utf-8"))
    meta["background_white_sigma_Lsun_pc2"] = white
    meta["background_white_mu_V"] = round(26.40 - 2.5 * np.log10(white), 2)
    meta_fn.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"  用时 {time.time() - t0:.1f}s；白点面亮度 {meta['background_white_mu_V']} mag/arcsec²")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="starfield_config.yaml")
    ap.add_argument("--mag-limit", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-background", action="store_true", help="跳过背景贴图")
    ap.add_argument("--bg-compute", type=int, nargs=2, default=[2048, 1024], metavar=("W", "H"),
                    help="背景积分分辨率（输出时再放大到配置的分辨率）")
    ap.add_argument("--background-only", action="store_true",
                    help="不重新生成：读 out/dso_raw.npz，只重画背景贴图")
    ap.add_argument("--overrides-only", action="store_true",
                    help="不重新采样：读 out/stars_base.npz（地标之后、覆盖之前的快照），重新应用 overrides.yaml 并导出")
    ap.add_argument("--export-only", action="store_true",
                    help="不重新生成：读 out/stars_raw.npz、dso_raw.npz，只重写事件、星表和观星器数据")
    args = ap.parse_args()

    t_all = time.time()
    cfg = load_config(args.config, mag_limit=args.mag_limit)
    out = Path(args.out or cfg["output"]["dir"])
    if args.background_only:
        dso = dict(np.load(out / "dso_raw.npz"))
        galaxy = Galaxy(cfg)
        dust = Dust(cfg, galaxy)
        render_and_save_background(cfg, galaxy, dust, dso, out, args.bg_compute)
        log(f"总用时 {time.time() - t_all:.1f}s")
        return
    if args.export_only:
        stars = dict(np.load(out / "stars_raw.npz"))
        dso = dict(np.load(out / "dso_raw.npz"))
        names = load_names(out / "names_raw.json")
        galaxy, clock = Galaxy(cfg), SkyClock(cfg)
        dust = Dust(cfg, galaxy)
        events, dso = timeline(cfg, galaxy, dust, clock, stars, dso)
        export(out, stars, dso, events, names, clock, cfg, keep_bg_meta=True)
        log(f"导出完成，用时 {time.time() - t_all:.1f}s")
        return
    if args.overrides_only:
        stars = dict(np.load(out / "stars_base.npz"))
        dso = dict(np.load(out / "dso_base.npz"))
        names = load_names(out / "names_base.json")
        old_rem = remnant_signature(dict(np.load(out / "dso_raw.npz"))) if (out / "dso_raw.npz").exists() else None
        galaxy, clock = Galaxy(cfg), SkyClock(cfg)
        dust = Dust(cfg, galaxy)
        log("[覆盖] 重新应用 overrides.yaml ...")
        stars, dso, events, names = after_landmarks(stars, dso, names, cfg, galaxy, dust, clock, out)
        export(out, stars, dso, events, names, clock, cfg, keep_bg_meta=True)
        if remnant_signature(dso) != old_rem and not args.no_background:
            log("  超新星遗迹有变化，重画背景")
            render_and_save_background(cfg, galaxy, dust, dso, out, args.bg_compute)
        log(f"完成，用时 {time.time() - t_all:.1f}s")
        return
    mag_lim = cfg["meta"]["generation_mag_limit"]
    for w in cfg["_warnings"]:
        log(f"[提示] {w}")
    log(f"生成深度 {mag_lim} 等；望远镜极限 {cfg['observing']['telescope']['limit_mag']} 等，"
        f"分辨率 {cfg['observing']['telescope']['resolution_arcsec']}″；宿 {cfg['sky_clock']['lunar_mansions']['count']} 个；"
        f"卫星推算岁差周期约 {cfg['_derived']['precession_period_estimate_yr']} 年")

    galaxy = Galaxy(cfg)
    dust = Dust(cfg, galaxy)
    clock = SkyClock(cfg)
    log(f"尘埃云 {len(dust.c_pos)} 个")

    log("[1/6] 采样场星 ...")
    stars, strata = generate_stars(cfg, galaxy, mag_lim, log)
    stars["cluster_id"] = np.full(stars["x"].size, -1, np.int32)

    log("[2/6] 星团、星协、深空天体 ...")
    t0 = time.time()
    members, dso = make_groups(cfg, galaxy, dust, IMF(cfg), mag_lim, log)
    stars = concat([stars] + members)
    log(f"  用时 {time.time() - t0:.1f}s")

    log("[3/6] 变星、伴星、食变星 ...")
    t0 = time.time()
    assign_pulsators(stars, cfg)
    add_companions(stars, cfg)
    assign_eclipsing(stars)
    log(f"  用时 {time.time() - t0:.1f}s")

    log("[4/6] 消光 ...")
    t0 = time.time()
    add_extinction(stars, dust)
    stars = subset(stars, stars["V"] <= mag_lim)
    log(f"  用时 {time.time() - t0:.1f}s；≤{mag_lim} 等的星 {stars['V'].size:,} 颗")

    log("[5/6] 地标约束 ...")
    stars, names = apply_landmarks(stars, dso, cfg, galaxy, dust, log)
    dso = finish_dso(dso, dust)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "stars_base.npz", **{k: v for k, v in stars.items() if isinstance(v, np.ndarray)})
    np.savez_compressed(out / "dso_base.npz", **dso)
    save_names(out / "names_base.json", names)

    log("[6/6] 手动覆盖、天球位置、双星可分辨性、事件时间线、超新星遗迹 ...")
    stars, dso, events, names = after_landmarks(stars, dso, names, cfg, galaxy, dust, clock, out)

    summary(stars, mag_lim)
    summary_extra(stars, dso, cfg)

    log("[导出] 星表与观星器数据 ...")
    t0 = time.time()
    export(out, stars, dso, events, names, clock, cfg)
    log(f"  用时 {time.time() - t0:.1f}s")

    if not args.no_background:
        render_and_save_background(cfg, galaxy, dust, dso, out, args.bg_compute)
    log(f"\n总用时 {time.time() - t_all:.1f}s")


def save_names(fn, names):
    # 键：(系统高位, 序号) 是系统名；(系统高位, 序号, 成员) 是成员自己的名字
    rows = [[str(int(k[0])), int(k[1]), (int(k[2]) if len(k) > 2 else -1), n] for k, n in names.items()]
    fn.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def load_names(fn):
    if not fn.exists():
        return {}
    out = {}
    for row in json.loads(fn.read_text(encoding="utf-8")):
        if len(row) == 3:
            h, i, n = row
            out[(int(h), int(i))] = n
        else:
            h, i, c, n = row
            out[(int(h), int(i)) if c < 0 else (int(h), int(i), int(c))] = n
    return out


def remnant_signature(dso):
    r = dso["kind"] == DSO_KINDS.index("remnant")
    return tuple(np.round(np.concatenate([dso["gal_lon"][r], dso["gal_lat"][r], dso["V_int"][r]]), 3).tolist())


def timeline(cfg, galaxy, dust, clock, stars, dso):
    """事件时间线（含覆盖层指定的超新星）+ 超新星遗迹接到深空天体表。"""
    events, sn_all = make_events(cfg, galaxy, dust, clock, log)
    events, sn_all = add_star_supernovae(events, sn_all, stars, cfg, clock)
    n_manual = sum(bool(e.get("manual")) for e in events)
    if n_manual:
        log(f"  覆盖层指定的超新星 {n_manual} 次")
    dso = add_remnants(dso, make_remnants(cfg, galaxy, sn_all, clock), dust, REMNANT_MAG_LIMIT)
    log(f"  超新星遗迹 {int((dso['kind'] == DSO_KINDS.index('remnant')).sum())} 个")
    return events, dso


def after_landmarks(stars, dso, names, cfg, galaxy, dust, clock, out):
    stars, names = apply_overrides(stars, names, cfg, galaxy, dust, log)
    stars.update(sky_positions(stars, clock, clock.ref_year))
    binary_resolvability(stars, cfg)
    events, dso = timeline(cfg, galaxy, dust, clock, stars, dso)
    np.savez_compressed(out / "stars_raw.npz", **{k: v for k, v in stars.items() if isinstance(v, np.ndarray)})
    np.savez_compressed(out / "dso_raw.npz", **dso)
    save_names(out / "names_raw.json", names)
    return stars, dso, events, names


def export(out, stars, dso, events, names, clock, cfg, keep_bg_meta=False):
    meta_fn = out / "viewer" / "sky_meta.json"
    old_meta = json.loads(meta_fn.read_text(encoding="utf-8")) if (keep_bg_meta and meta_fn.exists()) else {}
    write_all(out, stars, dso, events, names, clock, cfg, log)
    if old_meta:
        meta = json.loads(meta_fn.read_text(encoding="utf-8"))
        for k in ("background_white_sigma_Lsun_pc2", "background_white_mu_V"):
            if k in old_meta:
                meta[k] = old_meta[k]
        meta_fn.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
