"""决定论的计数器式随机数：一切随机量都由 (种子, 格子, 层, 序号, 属性号) 的哈希得到。

这样"看哪生成哪"：任何一颗星的属性只取决于它自己的键，与生成顺序、别处的参数无关。
"""
import numpy as np

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLD = np.uint64(0x9E3779B97F4A7C15)
_S30, _S27, _S31, _S11 = np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(11)


def to_u64(v):
    """任意整数数组（可为负）→ uint64（按位重解释）。"""
    a = np.asarray(v)
    if a.dtype == np.uint64:
        return a
    if a.dtype.kind == "u":
        return a.astype(np.uint64)
    return np.ascontiguousarray(a.astype(np.int64)).view(np.uint64)


def mix64(x):
    """splitmix64 终混函数。"""
    x = np.array(x, dtype=np.uint64, copy=True)
    with np.errstate(over="ignore"):
        x ^= x >> _S30
        x *= _M1
        x ^= x >> _S27
        x *= _M2
        x ^= x >> _S31
    return x


def combine(h, v):
    """把整数 v 混入哈希 h，返回新哈希（支持广播）。"""
    h = to_u64(h)
    with np.errstate(over="ignore"):
        return mix64(h ^ mix64(to_u64(v) + _GOLD))


def uniform(h, attr):
    """(0, 1) 内的均匀数；attr 为属性编号。"""
    x = combine(h, attr)
    return (x >> _S11).astype(np.float64) * (1.0 / 9007199254740992.0) + 2.0 ** -54


def normal(h, attr):
    """标准正态（Box-Muller，用 attr 与 attr+7919 两个均匀数）。"""
    u1 = uniform(h, attr)
    u2 = uniform(h, attr + 7919)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def poisson(lam, u, u_aux):
    """由均匀数决定论地得到泊松数。lam 小用逆累积分布，大用正态近似。"""
    lam = np.asarray(lam, dtype=np.float64)
    k = np.zeros(lam.shape, dtype=np.int64)
    small = lam < 40.0
    idx = np.nonzero(small)[0]
    if idx.size:
        l = lam[idx]
        uu = u[idx]
        p = np.exp(-l)
        cdf = p.copy()
        kk = np.zeros(idx.size, dtype=np.int64)
        act = np.nonzero(uu > cdf)[0]
        while act.size:
            kk[act] += 1
            p[act] *= l[act] / kk[act]
            cdf[act] += p[act]
            act = act[uu[act] > cdf[act]]
            if kk.max() > 200:
                break
        k[idx] = kk
    big = np.nonzero(~small)[0]
    if big.size:
        l = lam[big]
        z = np.sqrt(-2.0 * np.log(u[big])) * np.cos(2.0 * np.pi * u_aux[big])
        k[big] = np.maximum(0, np.rint(l + np.sqrt(l) * z)).astype(np.int64)
    return k


def seed_key(seed, *parts):
    """由总种子和若干整数派生一个子模块哈希（标量 uint64）。"""
    h = mix64(np.uint64(seed))
    for p in parts:
        h = combine(h, p)
    return h


def module_rng(seed, name):
    """给"只生成一次"的子模块（尘埃云、星团表等）一个普通的 numpy 生成器。"""
    code = int.from_bytes(name.encode("utf-8")[:8].ljust(8, b"\0"), "little")
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, code & 0xFFFFFFFF, code >> 32])
