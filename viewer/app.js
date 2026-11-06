// 观星器：读 ../out/viewer/ 下的数据，用 WebGL2 画星点与背景（球极投影）。
// 坐标链：固定黄道（数据）→ 当年黄道（岁差）→ 赤道（黄赤夹角）→ 地平（恒星时 + 纬度）→ 相机。
"use strict";

const DATA = new URLSearchParams(location.search).get("data") || "../out/viewer/";   // ?data=别的目录/ 可以看另一份输出
const D2R = Math.PI / 180;
const NF = 13; // 每颗星的浮点数
const VAR_CN = { DCEP: "经典造父变星", RRAB: "天琴 RR 型", DSCT: "盾牌 δ 型", BCEP: "仙王 β 型", M: "米拉型（长周期）",
  SR: "半规则变星", L: "不规则变星", EA: "大陵型食双星", EB: "天琴 β 型食双星", EW: "大熊 W 型食双星" };
const PHASE_CN = { MS: "主序", SG: "亚巨星", RGB: "红巨星支", CHeB: "氦燃烧（巨星/超巨星）", AGB: "渐近巨星支", WD: "白矮星" };
const RES_CN = { none: "分不开", telescope: "望远镜可分开", naked_eye: "肉眼可分开" };
const EV_CN = { supernova_Ia: "超新星（Ia）", supernova_CC: "超新星（核坍缩）", nova: "新星", great_comet: "大彗星" };

// ---------------- 小矩阵工具（行优先 3x3） ----------------
const M3 = {
  mul(a, b) { const r = new Array(9); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j]; return r; },
  T(a) { return [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]; },
  ap(a, v) { return [a[0] * v[0] + a[1] * v[1] + a[2] * v[2], a[3] * v[0] + a[4] * v[1] + a[5] * v[2], a[6] * v[0] + a[7] * v[1] + a[8] * v[2]]; },
  rx(d) { const c = Math.cos(d * D2R), s = Math.sin(d * D2R); return [1, 0, 0, 0, c, -s, 0, s, c]; },
  rz(d) { const c = Math.cos(d * D2R), s = Math.sin(d * D2R); return [c, -s, 0, s, c, 0, 0, 0, 1]; },
};
const unitv = (lon, lat) => [Math.cos(lat * D2R) * Math.cos(lon * D2R), Math.cos(lat * D2R) * Math.sin(lon * D2R), Math.sin(lat * D2R)];
const norm = v => { const l = Math.hypot(v[0], v[1], v[2]); return [v[0] / l, v[1] / l, v[2] / l]; };

// ---------------- 状态 ----------------
const S = {
  meta: null, n: 0, raw: null, info: null, dso: [], events: [],
  year: 0, day: 80, hour: 22, playing: false, speed: 60,
  mode: 0, // 0 肉眼 1 望远镜
  az: 180, alt: 35, fov: 100, // 视野中心（方位从北向东量）与水平视场
  bgGain: 1, starGain: 1, sel: -1, see: false, found: [], deaths: [],
  mats: null, mag: null, screen: null,
};

const gl = document.getElementById("gl").getContext("webgl2", { antialias: false, premultipliedAlpha: false });
const ov = document.getElementById("ov");
const ctx = ov.getContext("2d");
if (!gl) document.getElementById("load").textContent = "这个浏览器不支持 WebGL2";

// ---------------- 着色器 ----------------
const PROJ_GLSL = `
uniform mat3 u_h2c; uniform float u_scale; uniform vec2 u_center; uniform vec2 u_res;
vec2 projStereo(vec3 c) { float k = 2.0 / (1.0 + c.z); return vec2(c.x * k, c.y * k); }
`;
const VS_STAR = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_u; layout(location=1) in vec3 a_du; layout(location=2) in float a_mag; layout(location=3) in float a_bv;
uniform mat3 u_fe2h; uniform float u_kyr, u_lim, u_ext, u_gain, u_dpr, u_see;
${PROJ_GLSL}
out vec3 v_col; out float v_alpha;
vec3 bvColor(float bv) {
  bv = clamp(bv, -0.4, 2.0);
  float t = (bv + 0.4) / 2.4;
  vec3 hot = vec3(0.62, 0.74, 1.0), mid = vec3(1.0, 0.97, 0.92), cool = vec3(1.0, 0.62, 0.35);
  return t < 0.3 ? mix(hot, mid, t / 0.3) : mix(mid, cool, (t - 0.3) / 0.7);
}
void main() {
  vec3 d = normalize(a_u + a_du * u_kyr);
  vec3 h = u_fe2h * d;
  float sa = h.z;
  vec3 c = u_h2c * h;
  if ((sa < -0.01 && u_see < 0.5) || c.z < -0.95 || a_mag > 50.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
  float below = sa < 0.0 ? 1.0 : 0.0;
  sa = abs(sa);                          // 透视地面时，地下的星按镜像高度算大气消光
  float X = 1.0 / (sa + 0.025 * exp(-11.0 * sa));
  float m = a_mag + u_ext * (X - 1.0);
  float r = u_lim - m;
  if (r < 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
  vec2 p = projStereo(c) * u_scale;
  gl_Position = vec4(p / (u_res * 0.5), 0.0, 1.0);
  float f = pow(10.0, 0.4 * r);
  gl_PointSize = clamp((2.2 + 1.5 * pow(r, 1.2)) * u_gain, 2.0, 44.0) * u_dpr;
  v_alpha = clamp(0.35 + 0.22 * r, 0.35, 1.0);
  v_col = bvColor(a_bv) * min(1.6, 0.55 + 0.12 * f) * (1.0 - 0.45 * below);
}`;
const FS_STAR = `#version 300 es
precision highp float;
in vec3 v_col; in float v_alpha; out vec4 o;
void main() {
  vec2 q = gl_PointCoord * 2.0 - 1.0; float r2 = dot(q, q);
  if (r2 > 1.0) discard;
  float core = exp(-r2 * 5.5) + 0.25 * exp(-r2 * 1.6);
  o = vec4(v_col * core * v_alpha, 1.0);
}`;
const VS_QUAD = `#version 300 es
layout(location=0) in vec2 a; out vec2 v_px; uniform vec2 u_res;
void main() { v_px = (a * 0.5) * u_res; gl_Position = vec4(a, 0.0, 1.0); }`;
const FS_BG = `#version 300 es
precision highp float;
in vec2 v_px; out vec4 o;
uniform sampler2D u_tex; uniform mat3 u_c2g; uniform mat3 u_c2h; uniform float u_scale, u_gain, u_ext, u_sunAlt, u_moonGlow, u_mwVis, u_see;
void main() {
  vec2 P = v_px / u_scale; float r2 = dot(P, P);
  vec3 c = vec3(4.0 * P.x, 4.0 * P.y, 4.0 - r2) / (4.0 + r2);
  vec3 h = u_c2h * c;
  float gnd = 0.012 + 0.02 * exp(min(h.z, 0.0) * 6.0);
  vec3 ground = vec3(gnd * 0.8, gnd * 0.85, gnd);
  if (h.z < 0.0 && u_see < 0.5) { o = vec4(ground, 1.0); return; }   // 地面
  float hz = abs(h.z);
  vec3 g = u_c2g * c;
  float lon = atan(g.y, g.x); float lat = asin(clamp(g.z, -1.0, 1.0));
  vec2 uv = vec2(fract(lon / 6.2831853), 0.5 - lat / 3.1415927);
  vec3 mw = texture(u_tex, uv).rgb;
  mw = mw * mw;                         // PNG 做过开方压缩，大致还原成线性
  float X = 1.0 / (hz + 0.025 * exp(-11.0 * hz));
  float att = pow(10.0, -0.4 * u_ext * (X - 1.0));
  vec3 col = mw * 0.28 * u_gain * att * u_mwVis;    // 天光一亮，银河就淹没了
  // 气辉 + 地平附近的光
  col += vec3(0.010, 0.012, 0.018) * (0.6 + 0.6 * (1.0 - hz));
  // 暮光 / 白天
  float tw = smoothstep(-18.0, 0.0, u_sunAlt);
  float day = smoothstep(-6.0, 6.0, u_sunAlt);
  col += tw * vec3(0.10, 0.12, 0.22) * (1.0 - hz * 0.6) + day * vec3(0.30, 0.45, 0.80);
  col += u_moonGlow * vec3(0.05, 0.07, 0.11) * (1.0 - hz * 0.4);
  if (h.z < 0.0) col = mix(col * 0.55, ground * 2.0 + vec3(0.02, 0.025, 0.02), 0.5);   // 半透明地面
  o = vec4(col, 1.0);
}`;

function shader(type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function program(vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, shader(gl.VERTEX_SHADER, vs)); gl.attachShader(p, shader(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {}; const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i); u[a.name] = gl.getUniformLocation(p, a.name); }
  return { p, u };
}

let progStar, progBg, vaoStars, vaoEv, vaoQuad, bufMag, bufEv, texBg, evCount = 0;

// ---------------- 读数据 ----------------
async function load() {
  const [meta, bin] = await Promise.all([
    fetch(DATA + "sky_meta.json").then(r => r.json()),
    fetch(DATA + "stars.bin").then(r => r.arrayBuffer()),
  ]);
  S.meta = meta; S.raw = new Float32Array(bin); S.n = S.raw.length / NF;
  S.year = meta.reference_year;
  S.mag = new Float32Array(S.n);
  fetch(DATA + "stars_info.json").then(r => r.json()).then(j => { S.info = j; buildDeaths(); });
  fetch(DATA + "comps.json").then(r => r.ok ? r.json() : null).then(j => { S.comps = j; }).catch(() => {});
  fetch(DATA + "dso.json").then(r => r.json()).then(j => { S.dso = j; });
  fetch(DATA + "events.json").then(r => r.json()).then(j => { S.events = j; listEvents(); buildDeaths(); });
  const img = new Image(); img.src = DATA + (meta.background || "background.png");
  img.onload = () => uploadBg(img);
  initGL();
  updateMags(skyState());
  lastSig = `${S.year}|${S.day}|${S.hour.toFixed(4)}|${S.mode}`;
  initUI();
  document.getElementById("load").style.display = "none";
  requestAnimationFrame(frame);
}

function initGL() {
  progStar = program(VS_STAR, FS_STAR);
  progBg = program(VS_QUAD, FS_BG);
  // 星：静态 u, du, bv；动态 mag
  const st = new Float32Array(S.n * 7);
  for (let i = 0; i < S.n; i++) {
    const o = i * NF;
    st[i * 7] = S.raw[o]; st[i * 7 + 1] = S.raw[o + 1]; st[i * 7 + 2] = S.raw[o + 2];
    st[i * 7 + 3] = S.raw[o + 3]; st[i * 7 + 4] = S.raw[o + 4]; st[i * 7 + 5] = S.raw[o + 5];
    st[i * 7 + 6] = S.raw[o + 7];
  }
  vaoStars = gl.createVertexArray(); gl.bindVertexArray(vaoStars);
  const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, st, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 28, 12);
  gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 1, gl.FLOAT, false, 28, 24);
  bufMag = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, bufMag); gl.bufferData(gl.ARRAY_BUFFER, S.mag, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 4, 0);
  // 事件（客星、彗星）：每帧重建的小缓冲
  vaoEv = gl.createVertexArray(); gl.bindVertexArray(vaoEv);
  bufEv = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, bufEv);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 32, 0);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 32, 12);
  gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 32, 24);
  gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 1, gl.FLOAT, false, 32, 28);
  // 全屏四边形
  vaoQuad = gl.createVertexArray(); gl.bindVertexArray(vaoQuad);
  const q = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, q);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
  gl.bindVertexArray(null);
  // 背景占位（1 像素黑）
  texBg = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, texBg);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
}

function uploadBg(img) {
  const max = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  let src = img;
  if (img.width > max) {
    const c = document.createElement("canvas"); c.width = max; c.height = max / 2;
    c.getContext("2d").drawImage(img, 0, 0, c.width, c.height); src = c;
  }
  gl.bindTexture(gl.TEXTURE_2D, texBg);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, src);
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}

// ---------------- 天文 ----------------
function skyState() {
  const m = S.meta, yr = S.year;
  const dyr = yr - m.reference_year;
  const prec = 360 * dyr / m.precession_period_yr;
  const eps = m.obliquity_deg + (m.obliquity_osc_amp_deg || 0) * Math.sin(2 * Math.PI * dyr / (m.obliquity_osc_period_yr || 41000));
  const Mfe2ed = M3.rz(prec - m.equinox_ecl_lon_at_reference_deg); // 固定黄道 → 当年黄道
  const Med2eq = M3.rx(eps);
  // 太阳：春分约在第 80 天
  const lamSun = 360 * (S.day - 80) / m.year_days;
  const sunEcl = unitv(lamSun, 0);
  const sunEq = M3.ap(Med2eq, sunEcl);
  const raSun = Math.atan2(sunEq[1], sunEq[0]) / D2R;
  const lst = raSun + (S.hour - 12) * 15;
  const phi = m.latitude_deg * D2R;
  const Mloc = [0, 1, 0, -Math.sin(phi), 0, Math.cos(phi), Math.cos(phi), 0, Math.sin(phi)];
  const Meq2h = M3.mul(Mloc, M3.rz(-lst));
  const Med2h = M3.mul(Meq2h, Med2eq);
  const Mfe2h = M3.mul(Med2h, Mfe2ed);
  const Mg2h = M3.mul(Mfe2h, m.gal_to_ecl.flat());
  const sunH = M3.ap(Med2h, sunEcl);
  // 月亮（主卫星）：沿白道匀速，交点退行
  const moon = m.moons && m.moons[0];
  let moonH = null, moonIllum = 0, moonEclD = null;
  if (moon) {
    const tDays = dyr * m.year_days + S.day + S.hour / 24;
    const lam = (218 + 360 * tDays / moon.sidereal_period_days) % 360;
    const node = 125 - 360 * tDays / (moon.node_regression_period_yr * m.year_days);
    const beta = moon.inclination_to_ecliptic_deg * Math.sin((lam - node) * D2R);
    moonEclD = unitv(lam, beta);
    moonH = M3.ap(Med2h, moonEclD);
    const elong = Math.acos(Math.max(-1, Math.min(1, moonEclD[0] * sunEcl[0] + moonEclD[1] * sunEcl[1] + moonEclD[2] * sunEcl[2])));
    moonIllum = (1 - Math.cos(elong)) / 2;
  }
  return { prec, eps, lst, Mfe2h, Mg2h, Meq2h, Med2h, sunH, sunAlt: Math.asin(sunH[2]) / D2R, moonH, moonIllum, lamSun, kyr: dyr / 1000 };
}

function camera() {
  const az = S.az * D2R, al = S.alt * D2R;
  const f = [Math.sin(az) * Math.cos(al), Math.cos(az) * Math.cos(al), Math.sin(al)];
  let r = [Math.cos(az), -Math.sin(az), 0];
  const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
  return [r[0], r[1], r[2], u[0], u[1], u[2], f[0], f[1], f[2]]; // 行：右、上、前（地平 → 相机）
}

function viewScale() {
  return (ov.width * 0.5) / (2 * Math.tan(S.fov * D2R / 4));
}

// 地平向量 → 屏幕像素（设备像素），不可见返回 null
function projectH(h, Mh2c, scale) {
  const c = M3.ap(Mh2c, h);
  if (c[2] < -0.9) return null;
  const k = 2 / (1 + c[2]);
  return [ov.width / 2 + c[0] * k * scale, ov.height / 2 - c[1] * k * scale];
}

// ---------------- 变星 ----------------
function varShape(type, ph, t, ep) {
  switch (type) {
    case 1: case 2: { // 造父、天琴 RR：快升慢降
      return ph < 0.15 ? 1 - ph / 0.15 : (ph - 0.15) / 0.85; }
    case 3: case 4: case 5: return 0.5 - 0.5 * Math.cos(2 * Math.PI * ph);
    case 6: case 7: { // 半规则 / 不规则
      const x = t / 37 + ep * 100;
      return 0.5 + 0.25 * Math.sin(x) + 0.15 * Math.sin(x * 2.7 + 1.3) + 0.1 * Math.sin(x * 0.31 + 4.0); }
    case 8: { // 大陵型：主食 + 浅的次食
      const a = Math.min(ph, 1 - ph), b = Math.abs(ph - 0.5);
      return a < 0.04 ? 1 - a / 0.04 : (b < 0.04 ? 0.3 * (1 - b / 0.04) : 0); }
    case 9: return Math.pow(Math.abs(Math.cos(2 * Math.PI * ph)), 0.7) * (ph < 0.25 || ph > 0.75 ? 1 : 0.6);
    case 10: return 0.5 + 0.5 * Math.cos(4 * Math.PI * ph);
    default: return 0;
  }
}

function updateMags(st) {
  const r = S.raw, n = S.n;
  const tDays = (S.year - S.meta.reference_year) * S.meta.year_days + S.day + S.hour / 24;
  const naked = S.mode === 0;
  for (let i = 0; i < n; i++) {
    const o = i * NF;
    let m = naked ? r[o + 12] : r[o + 6];
    const amp = r[o + 8];
    if (amp > 0 && m < 50) {
      const per = r[o + 9], type = r[o + 11], ep = r[o + 10];
      const ph = per > 0 ? ((ep + tDays / per) % 1 + 1) % 1 : 0;
      m += amp * varShape(type, ph, tDays, ep);
    }
    S.mag[i] = m;
  }
  for (const x of S.deaths) if (tDays >= x.t) S.mag[x.i] = 99;   // 已经爆发成超新星的前身星
  gl.bindBuffer(gl.ARRAY_BUFFER, bufMag);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, S.mag);
}

// ---------------- 事件 ----------------
// 覆盖层指定爆发的星：事件里带 star_id，爆发那天起前身星从天上消失
function buildDeaths() {
  if (!S.info || !S.events.length) return;
  const sysid = S.info.sys_id || S.info.id, idx = new Map();
  sysid.forEach((id, i) => { if (!S.info.is_companion[i] && !idx.has(id)) idx.set(id, i); });
  S.deaths = S.events.filter(e => e.star_id && idx.has(e.star_id)).map(e => ({
    i: idx.get(e.star_id), year: e.year, t: (e.year - S.meta.reference_year) * S.meta.year_days + (e.day || 0) }));
  lastSig = "";
}

function activeEvents() {
  const out = [];
  const tNow = S.year * S.meta.year_days + S.day;
  for (const e of S.events) {
    if (e.year === undefined) continue;
    const dt = tNow - (e.year * S.meta.year_days + (e.day || 0));
    if (dt < 0 || dt > (e.visible_days || 30)) continue;
    let u;
    if (e.gal_lon !== undefined) u = M3.ap(S.meta.gal_to_ecl.flat(), unitv(e.gal_lon, e.gal_lat));
    else if (e.ecl_lon !== undefined) { // 彗星：给的是当年黄道坐标，换回固定黄道
      const prec = 360 * (e.year - S.meta.reference_year) / S.meta.precession_period_yr - S.meta.equinox_ecl_lon_at_reference_deg;
      u = M3.ap(M3.T(M3.rz(prec)), unitv(e.ecl_lon, e.ecl_lat));
    } else continue;
    const mag = e.peak_mag + (6.5 - e.peak_mag) * Math.max(0, dt) / (e.visible_days || 30);
    out.push({ e, u, mag });
  }
  return out;
}

function listEvents() {
  const el = document.getElementById("events");
  const near = S.events.filter(e => e.type !== "great_comet" && e.peak_mag <= 2).slice(0, 400);
  el.innerHTML = "<div style='color:#e8c48a'>亮于 2 等的客星（点击跳转）：</div>" +
    near.map((e, k) => `<div data-k="${S.events.indexOf(e)}">${e.year} 年 第${e.day}天 · ${EV_CN[e.type] || e.type} · ${e.peak_mag} 等${e.observable ? "" : "（此地不可见）"}${e.future ? "（未来）" : ""}</div>`).join("");
  el.onclick = ev => {
    const k = ev.target.dataset.k; if (k === undefined) return;
    const e = S.events[+k];
    S.year = e.year; S.day = Math.max(1, (e.day || 0) + 2); syncUI();
    // 把视野转到它
    const st = skyState();
    const u = M3.ap(S.meta.gal_to_ecl.flat(), unitv(e.gal_lon, e.gal_lat));
    const h = M3.ap(st.Mfe2h, u);
    S.az = (Math.atan2(h[0], h[1]) / D2R + 360) % 360; S.alt = Math.max(5, Math.asin(h[2]) / D2R);
  };
}

// ---------------- 绘制 ----------------
let lastSig = "";
function frame() {
  resize();
  if (S.playing) {
    S.hour += S.speed / 3600;
    while (S.hour >= 24) { S.hour -= 24; S.day += 1; }
    if (S.day > Math.floor(S.meta.year_days)) { S.day = 1; S.year += 1; }
    syncUI();
  }
  const st = skyState();
  const sig = `${S.year}|${S.day}|${S.hour.toFixed(4)}|${S.mode}`;
  if (sig !== lastSig) { updateMags(st); lastSig = sig; }
  const Mh2c = camera();
  const scale = viewScale();
  const dpr = window.devicePixelRatio || 1;

  // 有效极限星等：暮光、月光
  let lim = S.mode === 0 ? S.meta.naked_eye_limit : S.meta.telescope_limit;
  const sunPen = Math.max(0, Math.min(12, (st.sunAlt + 18) * 0.75));
  let moonGlow = 0;
  if (document.getElementById("cMoon").checked && st.moonH && st.moonH[2] > 0) moonGlow = st.moonIllum * Math.sqrt(st.moonH[2]);
  lim -= sunPen + 1.6 * moonGlow;
  const ext = document.getElementById("cExt").checked ? S.meta.extinction_per_airmass : 0;

  gl.viewport(0, 0, ov.width, ov.height);
  gl.disable(gl.BLEND);
  gl.useProgram(progBg.p);
  const Mc2h = M3.T(Mh2c);
  gl.uniformMatrix3fv(progBg.u.u_c2h, true, Mc2h);
  gl.uniformMatrix3fv(progBg.u.u_c2g, true, M3.mul(M3.T(st.Mg2h), Mc2h));
  gl.uniform2f(progBg.u.u_res, ov.width, ov.height);
  gl.uniform1f(progBg.u.u_scale, scale);
  gl.uniform1f(progBg.u.u_gain, S.bgGain * (S.mode === 1 ? 0.6 : 1));
  gl.uniform1f(progBg.u.u_ext, ext);
  gl.uniform1f(progBg.u.u_sunAlt, st.sunAlt);
  gl.uniform1f(progBg.u.u_moonGlow, moonGlow);
  const smooth = (a, b, x) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
  gl.uniform1f(progBg.u.u_mwVis, (1 - smooth(-18, -10, st.sunAlt)) / (1 + 3 * moonGlow));
  gl.uniform1f(progBg.u.u_see, S.see ? 1 : 0);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texBg); gl.uniform1i(progBg.u.u_tex, 0);
  gl.bindVertexArray(vaoQuad); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
  gl.useProgram(progStar.p);
  gl.uniformMatrix3fv(progStar.u.u_fe2h, true, st.Mfe2h);
  gl.uniformMatrix3fv(progStar.u.u_h2c, true, Mh2c);
  gl.uniform1f(progStar.u.u_kyr, st.kyr);
  gl.uniform1f(progStar.u.u_lim, lim);
  gl.uniform1f(progStar.u.u_ext, ext);
  gl.uniform1f(progStar.u.u_gain, S.starGain * (S.mode === 1 ? 0.85 : 1));
  gl.uniform1f(progStar.u.u_dpr, dpr);
  gl.uniform1f(progStar.u.u_scale, scale);
  gl.uniform2f(progStar.u.u_res, ov.width, ov.height);
  gl.uniform1f(progStar.u.u_see, S.see ? 1 : 0);
  gl.bindVertexArray(vaoStars); gl.drawArrays(gl.POINTS, 0, S.n);

  // 客星 / 彗星
  const evs = document.getElementById("cEv").checked ? activeEvents() : [];
  if (evs.length) {
    const a = new Float32Array(evs.length * 8);
    evs.forEach((x, k) => { a.set([x.u[0], x.u[1], x.u[2], 0, 0, 0, x.mag, x.e.type === "great_comet" ? 0.6 : 0.2], k * 8); });
    gl.bindBuffer(gl.ARRAY_BUFFER, bufEv); gl.bufferData(gl.ARRAY_BUFFER, a, gl.DYNAMIC_DRAW);
    gl.uniform1f(progStar.u.u_kyr, 0);
    gl.bindVertexArray(vaoEv); gl.drawArrays(gl.POINTS, 0, evs.length);
  }
  gl.bindVertexArray(null);

  drawOverlay(st, Mh2c, scale, lim, evs);
  status(st, lim);
  requestAnimationFrame(frame);
}

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.floor(innerWidth * dpr), h = Math.floor(innerHeight * dpr);
  for (const c of [gl.canvas, ov]) if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
}

function polyline(pts) {
  ctx.beginPath(); let pen = false, prev = null;
  for (const p of pts) {
    if (!p || (prev && Math.hypot(p[0] - prev[0], p[1] - prev[1]) > ov.width * 0.3)) { pen = false; prev = p; continue; }
    if (!pen) { ctx.moveTo(p[0], p[1]); pen = true; } else ctx.lineTo(p[0], p[1]);
    prev = p;
  }
  ctx.stroke();
}

function drawOverlay(st, Mh2c, scale, lim, evs) {
  const dpr = window.devicePixelRatio || 1;
  ctx.clearRect(0, 0, ov.width, ov.height);
  ctx.lineWidth = 1 * dpr;
  ctx.font = `${12 * dpr}px system-ui, "Microsoft YaHei", sans-serif`;
  const P = h => projectH(h, Mh2c, scale);
  const lo = S.see ? -1.1 : -0.05;          // 透视地面时，地平线下的东西也画
  const altaz = (az, al) => [Math.sin(az * D2R) * Math.cos(al * D2R), Math.cos(az * D2R) * Math.cos(al * D2R), Math.sin(al * D2R)];

  if (document.getElementById("cGrid").checked) {
    ctx.strokeStyle = "rgba(120,170,120,0.28)";
    for (let al = S.see ? -75 : 15; al < 90; al += 15) if (al) polyline(Array.from({ length: 181 }, (_, i) => P(altaz(i * 2, al))));
    for (let az = 0; az < 360; az += 30) polyline(Array.from({ length: S.see ? 91 : 46 }, (_, i) => P(altaz(az, S.see ? i * 2 - 90 : i * 2))));
  }
  if (document.getElementById("cEq").checked) {
    ctx.strokeStyle = "rgba(110,150,220,0.3)";
    const Me = st.Meq2h;
    for (let d = -60; d <= 75; d += 15) polyline(Array.from({ length: 181 }, (_, i) => { const h = M3.ap(Me, unitv(i * 2, d)); return h[2] > lo ? P(h) : null; }));
    for (let ra = 0; ra < 360; ra += 30) polyline(Array.from({ length: 61 }, (_, i) => { const h = M3.ap(Me, unitv(ra, -60 + i * 2.5)); return h[2] > lo ? P(h) : null; }));
    const pole = P(M3.ap(Me, [0, 0, 1])); if (pole) { ctx.fillStyle = "rgba(150,180,255,0.8)"; ctx.fillText("天极", pole[0] + 4, pole[1] - 4); }
  }
  if (document.getElementById("cEcl").checked) {
    ctx.strokeStyle = "rgba(230,190,110,0.35)";
    polyline(Array.from({ length: 361 }, (_, i) => { const h = M3.ap(st.Med2h, unitv(i, 0)); return h[2] > lo ? P(h) : null; }));
  }
  // 地平线与方位
  ctx.strokeStyle = "rgba(200,200,200,0.35)";
  polyline(Array.from({ length: 361 }, (_, i) => P(altaz(i, 0))));
  ctx.fillStyle = "rgba(230,210,170,0.9)";
  for (const [az, t] of [[0, "北"], [90, "东"], [180, "南"], [270, "西"]]) { const p = P(altaz(az, 1)); if (p) ctx.fillText(t, p[0] - 6 * dpr, p[1] - 4 * dpr); }

  // 月亮
  if (document.getElementById("cMoon").checked && st.moonH && st.moonH[2] > (S.see ? -1.1 : -0.01)) {
    const p = P(st.moonH);
    if (p) {
      const rad = Math.max(3 * dpr, 0.26 * D2R * 2 * scale);
      const sp = P(st.sunH) || [p[0] + 1, p[1]];
      const ang = Math.atan2(sp[1] - p[1], sp[0] - p[0]);
      ctx.save(); ctx.translate(p[0], p[1]); ctx.rotate(ang);
      ctx.fillStyle = "rgba(40,40,45,0.9)"; ctx.beginPath(); ctx.arc(0, 0, rad, 0, 2 * Math.PI); ctx.fill();
      const k = 1 - 2 * st.moonIllum; // 明暗界线椭圆
      ctx.fillStyle = "rgba(245,240,225,0.97)"; ctx.beginPath();
      ctx.arc(0, 0, rad, -Math.PI / 2, Math.PI / 2, false);
      ctx.ellipse(0, 0, Math.abs(k) * rad, rad, 0, Math.PI / 2, -Math.PI / 2, k > 0);
      ctx.fill(); ctx.restore();
    }
  }
  // 深空天体
  if (document.getElementById("cDso").checked) {
    ctx.strokeStyle = "rgba(160,200,255,0.55)"; ctx.fillStyle = "rgba(160,200,255,0.75)";
    const Mg = st.Mfe2h;
    for (const d of S.dso) {
      if (d.V > (S.mode === 0 ? 6 : 9)) continue;
      const h = M3.ap(Mg, d.u); if (h[2] < lo + 0.05) continue;
      const p = P(h); if (!p) continue;
      const r = Math.max(4 * dpr, d.size_deg * 0.5 * D2R * 2 * scale);
      ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 2 * Math.PI); ctx.stroke();
      ctx.fillText(d.name, p[0] + r + 2, p[1]);
    }
  }
  // 客星标签
  ctx.fillStyle = "rgba(255,170,120,0.9)";
  for (const x of evs) { const h = M3.ap(st.Mfe2h, x.u); if (h[2] < lo + 0.05) continue; const p = P(h); if (p) ctx.fillText(`${EV_CN[x.e.type] || x.e.type} ${x.mag.toFixed(1)}等`, p[0] + 8, p[1] - 8); }
  // 亮星编号
  if (document.getElementById("cLab").checked && S.info) {
    ctx.fillStyle = "rgba(200,210,230,0.6)";
    const pts = screenStars(st, Mh2c, scale, lim, Math.min(lim, S.mode === 0 ? 2.5 : 6));
    for (const s of pts) ctx.fillText(S.info.id[s.i].slice(0, 6), s.x + 5, s.y - 3);
  }
  // 搜索结果
  if (S.found.length && document.getElementById("qMark").checked) {
    ctx.strokeStyle = "rgba(120,220,255,0.7)";
    const lo2 = S.see ? -1.1 : 0, s = 5 * dpr;
    for (const i of S.found.slice(0, 3000)) {
      const o = i * NF;
      const u = norm([S.raw[o] + S.raw[o + 3] * st.kyr, S.raw[o + 1] + S.raw[o + 4] * st.kyr, S.raw[o + 2] + S.raw[o + 5] * st.kyr]);
      const h = M3.ap(st.Mfe2h, u); if (h[2] < lo2) continue;
      const p = P(h); if (!p || p[0] < 0 || p[1] < 0 || p[0] > ov.width || p[1] > ov.height) continue;
      ctx.strokeRect(p[0] - s, p[1] - s, 2 * s, 2 * s);
    }
  }
  // 选中标记
  if (S.sel >= 0) {
    const o = S.sel * NF;
    const u = norm([S.raw[o] + S.raw[o + 3] * st.kyr, S.raw[o + 1] + S.raw[o + 4] * st.kyr, S.raw[o + 2] + S.raw[o + 5] * st.kyr]);
    const p = P(M3.ap(st.Mfe2h, u));
    if (p) { ctx.strokeStyle = "rgba(232,196,138,0.9)"; ctx.beginPath(); ctx.arc(p[0], p[1], 9 * dpr, 0, 2 * Math.PI); ctx.stroke(); }
  }
}

function screenStars(st, Mh2c, scale, lim, magCut) {
  const out = [];
  const M = M3.mul(Mh2c, st.Mfe2h);
  const ext = document.getElementById("cExt").checked ? S.meta.extinction_per_airmass : 0;
  for (let i = 0; i < S.n; i++) {
    if (S.mag[i] > magCut + 1.5) continue;
    const o = i * NF;
    const u = [S.raw[o] + S.raw[o + 3] * st.kyr, S.raw[o + 1] + S.raw[o + 4] * st.kyr, S.raw[o + 2] + S.raw[o + 5] * st.kyr];
    const h = M3.ap(st.Mfe2h, u); const l = Math.hypot(h[0], h[1], h[2]); const sa = h[2] / l;
    if (sa < -0.01 && !S.see) continue;
    const sz = Math.abs(sa);
    const X = 1 / (sz + 0.025 * Math.exp(-11 * sz));
    const m = S.mag[i] + ext * (X - 1);
    if (m > magCut) continue;
    const c = M3.ap(M, u).map(x => x / l);
    if (c[2] < -0.9) continue;
    const k = 2 / (1 + c[2]);
    const x = ov.width / 2 + c[0] * k * scale, y = ov.height / 2 - c[1] * k * scale;
    if (x < 0 || y < 0 || x > ov.width || y > ov.height) continue;
    out.push({ i, x, y, m });
  }
  return out;
}

function status(st, lim) {
  const season = ["春", "夏", "秋", "冬"][Math.floor((((st.lamSun % 360) + 360) % 360) / 90)];
  document.getElementById("status").textContent =
    `视场 ${S.fov.toFixed(S.fov < 2 ? 2 : 0)}°  中心 方位${S.az.toFixed(0)}° 高度${S.alt.toFixed(0)}°\n` +
    `太阳黄经 ${(((st.lamSun % 360) + 360) % 360).toFixed(0)}°（${season}）  太阳高度 ${st.sunAlt.toFixed(0)}°\n` +
    `岁差 ${st.prec.toFixed(1)}°  黄赤夹角 ${st.eps.toFixed(2)}°  当前极限 ${lim.toFixed(1)} 等` +
    (st.moonH ? `\n月相 ${(st.moonIllum * 100).toFixed(0)}%  月亮高度 ${(Math.asin(st.moonH[2]) / D2R).toFixed(0)}°` : "");
}

// ---------------- 点选 ----------------
function pick(px, py) {
  const st = skyState(); const Mh2c = camera(); const scale = viewScale();
  let lim = S.mode === 0 ? S.meta.naked_eye_limit : S.meta.telescope_limit;
  const pts = screenStars(st, Mh2c, scale, lim, lim);
  const dpr = window.devicePixelRatio || 1;
  let best = -1, bs = 1e9;
  for (const s of pts) {
    const d = Math.hypot(s.x - px * dpr, s.y - py * dpr);
    if (d > 14 * dpr) continue;
    const score = d + s.m * 2.5 * dpr;
    if (score < bs) { bs = score; best = s.i; }
  }
  S.sel = best;
  showInfo(best, st);
}

function fmtNum(x, d) { return x === null || x === undefined ? "—" : (+x).toFixed(d); }
function fmtLum(x) { return x >= 1000 ? x.toExponential(2).replace("e+", "×10^") : x >= 1 ? x.toFixed(1) : x.toPrecision(2); }
function fmtSep(x) { return x === null || x < 0 ? "—" : x < 1 ? `${x.toFixed(3)}″` : x < 60 ? `${x.toFixed(1)}″` : `${(x / 60).toFixed(1)}′`; }
function fmtP(d) { return d === null ? "—" : d < 2 ? `${(d * 24).toFixed(1)} 小时` : d < 800 ? `${d.toFixed(1)} 天` : `${(d / 365.25).toFixed(d < 3.6e4 ? 1 : 0)} 年`; }

function showInfo(i, st) {
  const box = document.getElementById("info");
  if (i < 0 || !S.info) { box.style.display = "none"; return; }
  const I = S.info, o = i * NF;
  const vt = S.raw[o + 11] | 0, vtn = I.var_types[vt];
  const ly = I.dist_pc[i] * 3.2616;
  const multi = I.mult[i] > 1;
  const unresolvedSys = multi && !I.is_companion[i] && I.id[i] === (I.sys_id ? I.sys_id[i] : I.id[i]);
  const title = I.name && I.name[i] ? I.name[i] : (multi ? (unresolvedSys ? `${["", "", "双星", "三合星"][I.mult[i]]}系统` : `${I.mult[i] === 2 ? "双星" : "三合星"}成员`) : "单星");
  const rows = [
    ["编号", I.id[i]],
    ["视星等", `${S.mag[i] >= 99 ? "—" : S.mag[i].toFixed(2)}（亮时 ${(S.mode === 0 ? S.raw[o + 12] : S.raw[o + 6]).toFixed(2)}）${unresolvedSys ? "，系统合成" : ""}`],
  ];
  if (unresolvedSys) rows.push(["合成光谱", I.spt_sys ? I.spt_sys[i] : I.spt[i]]);
  rows.push(
    [multi ? "这颗星" : "光谱型", multi ? `${I.spt[i] || "—"}（${unresolvedSys ? "主星 A" : I.id[i].slice(-1)}）` : (I.spt[i] || "—")],
    ["颜色 B−V", S.raw[o + 7].toFixed(2)],
    ["距离", `${I.dist_pc[i].toFixed(1)} pc（${ly.toFixed(ly < 100 ? 1 : 0)} 光年）`],
    ["视差", `${I.parallax_mas[i].toFixed(2)} 毫角秒`],
    ["绝对星等", `${I.MV[i] >= 99 ? "—" : I.MV[i].toFixed(2)}${unresolvedSys && I.MV_sys ? `（系统 ${I.MV_sys[i].toFixed(2)}）` : ""}`],
    ["质量", `${I.mass[i]} M☉`], ["年龄", I.age_myr[i] >= 1000 ? `${(I.age_myr[i] / 1000).toFixed(2)} Gyr` : `${I.age_myr[i]} Myr`],
    ["演化阶段", PHASE_CN[I.phase[i]] || I.phase[i]], ["表面温度", `${I.teff[i]} K`],
  );
  if (I.lum) rows.push(["光度 / 半径", `${fmtLum(I.lum[i])} L☉ · ${I.rad[i] < 0.1 ? I.rad[i].toPrecision(2) : I.rad[i].toPrecision(3)} R☉`]);
  rows.push(["消光", `${I.A_V[i]} 等`]);
  if (vt > 0) rows.push(["变星", `${VAR_CN[vtn] || vtn}，周期 ${S.raw[o + 9] > 0 ? S.raw[o + 9].toFixed(S.raw[o + 9] < 10 ? 3 : 1) + " 天" : "不定"}，变幅 ${S.raw[o + 8].toFixed(2)} 等`]);
  if (I.cluster[i]) rows.push(["所属", I.cluster[i]]);
  const death = S.deaths.find(x => x.i === i);
  if (death) rows.push(["结局", `${death.year} 年爆发为超新星${S.mag[i] >= 99 ? "（已爆发，现在看不到）" : ""}`]);
  // 多星系统：每颗成员自己的数据
  let sysHtml = "";
  const C = S.comps, cs = I.comp_start ? I.comp_start[i] : -1;
  if (multi && C && cs >= 0) {
    const n = I.comp_n[i];
    const tr = [];
    for (let k = cs; k < cs + n; k++) {
      const cur = C.entry[k] === i && (C.letter[k] === "A" ? !I.is_companion[i] : I.id[i].endsWith(C.letter[k]));
      const orbit = C.letter[k] === "A" ? "" : `${C.letter[k] === "B" ? "绕 A" : "绕 A+B"} · ${fmtP(C.period_d[k])} · ${fmtNum(C.a_au[k], C.a_au[k] < 1 ? 3 : 1)} AU · 相距 ${fmtSep(C.sep[k])} · ${RES_CN[C.res[k]]}`;
      tr.push(`<tr data-ent="${C.entry[k]}" class="${cur ? "cur" : ""}"><td><b>${C.letter[k]}</b> ${C.name[k] || ""}</td>` +
        `<td>${C.spt[k]} · ${fmtNum(C.mass[k], 2)} M☉ · ${C.teff[k]} K · ${fmtLum(C.lum[k])} L☉ · 自身 ${C.V[k] === null ? "看不见" : C.V[k].toFixed(2) + " 等"}` +
        `${C.var[k] ? " · " + (VAR_CN[I.var_types[C.var[k]]] || I.var_types[C.var[k]]) : ""}${orbit ? `<br><span class="dim">${orbit}</span>` : ""}</td></tr>`);
    }
    sysHtml = `<h3>系统成员</h3><table class="members">${tr.join("")}</table>` +
      `<div class="note">A 是原本最重、最先演化的那颗；它先变成白矮星或巨星时可能比 B 暗。点一行：分得开的成员会选中它自己。</div>`;
  } else if (multi) {
    rows.push(["伴星角距", I.sep_arcsec[i] >= 0 ? `${fmtSep(I.sep_arcsec[i])} · ${RES_CN[I.resolvable[i]]}` : "—"]);
  }
  box.innerHTML = `<span class="close" onclick="S.sel=-1;this.parentNode.style.display='none'">✕</span><h2>${title}</h2><table>` +
    rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join("") + "</table>" + sysHtml +
    `<div class="note">这是"宇宙里实际有什么"。伽利略时代的天文学家测不出视差、距离和光谱，只能目估亮度和颜色。</div>`;
  box.querySelectorAll("tr[data-ent]").forEach(tr => tr.onclick = () => {
    const ent = +tr.dataset.ent; S.sel = ent; showInfo(ent, skyState());
  });
  box.style.display = "block";
}

// ---------------- 界面 ----------------
function syncUI() {
  const $ = id => document.getElementById(id);
  $("year").value = S.year; $("yearR").value = S.year;
  $("day").value = S.day; $("dayv").textContent = `第 ${S.day} 天`;
  $("hour").value = S.hour;
  const h = Math.floor(S.hour), mi = Math.floor((S.hour - h) * 60);
  $("hourv").textContent = `${String(h).padStart(2, "0")}:${String(mi).padStart(2, "0")}`;
  $("bgv").textContent = S.bgGain.toFixed(2); $("sgv").textContent = S.starGain.toFixed(2);
}

function initUI() {
  const $ = id => document.getElementById(id);
  $("latv").textContent = S.meta.latitude_deg;
  $("day").max = Math.floor(S.meta.year_days);
  syncUI();
  $("year").onchange = e => { S.year = Math.round(+e.target.value); syncUI(); };
  $("yearR").oninput = e => { S.year = +e.target.value; syncUI(); };
  $("day").oninput = e => { S.day = +e.target.value; syncUI(); };
  $("hour").oninput = e => { S.hour = +e.target.value; syncUI(); };
  $("bgGain").oninput = e => { S.bgGain = +e.target.value; syncUI(); };
  $("starGain").oninput = e => { S.starGain = +e.target.value; syncUI(); };
  $("play").onclick = () => { S.playing = !S.playing; $("play").textContent = S.playing ? "❚❚ 暂停" : "▶ 播放"; $("play").classList.toggle("on", S.playing); };
  document.querySelectorAll("[data-speed]").forEach(b => b.onclick = () => {
    S.speed = +b.dataset.speed; if (!S.playing) $("play").click(); });
  const toggleSearch = initSearch();
  $("cSee").onchange = e => { S.see = e.target.checked; if (!S.see) S.alt = Math.max(-10, S.alt); };
  const setMode = m => {
    S.mode = m; $("mNaked").classList.toggle("on", m === 0); $("mTel").classList.toggle("on", m === 1);
    S.fov = m === 1 ? 2 : 100; lastSig = "";
    if (S.sel >= 0) showInfo(S.sel, skyState());
  };
  $("mNaked").onclick = () => setMode(0);
  $("mTel").onclick = () => setMode(1);
  const dirs = [[180, "南"], [270, "西"], [0, "北"], [90, "东"]]; let di = 0;
  $("look").onclick = () => { di = (di + 1) % 4; S.az = dirs[di][0]; S.alt = 35; $("look").textContent = "朝向：" + dirs[di][1]; };

  // 拖动视角、滚轮缩放、点击选星
  const gc = gl.canvas; let drag = null;
  gc.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY, az: S.az, alt: S.alt, moved: false }; gc.setPointerCapture(e.pointerId); });
  gc.addEventListener("pointermove", e => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    const degPerPx = S.fov / innerWidth;
    S.az = ((drag.az - dx * degPerPx / Math.max(0.2, Math.cos(S.alt * D2R))) % 360 + 360) % 360;
    S.alt = Math.max(S.see ? -90 : -10, Math.min(90, drag.alt + dy * degPerPx));
  });
  gc.addEventListener("pointerup", e => { if (drag && !drag.moved) pick(e.clientX, e.clientY); drag = null; });
  gc.addEventListener("wheel", e => {
    e.preventDefault();
    S.fov = Math.max(0.2, Math.min(200, S.fov * Math.exp(e.deltaY * 0.0012)));
  }, { passive: false });
  window.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT") return;
    if (e.key === " ") { $("play").click(); e.preventDefault(); }
    if (e.key === "t") setMode(S.mode ? 0 : 1);
    if (e.key === "g") $("cSee").click();
    if (e.key === "f") { toggleSearch(); e.preventDefault(); }
  });
}


// ---------------- 恒星搜索 ----------------
const LUM_GROUP = l => l === "V" ? "V" : l === "IV" ? "IV" : l === "III" ? "III" : l === "II" ? "II" : l.startsWith("I") ? "I" : "";
// 搜索的对象是单颗恒星：单星就是条目本身；多星系统的每个成员各算一颗（数据来自 comps.json）
function searchUnits() {
  if (S._units) return S._units;
  const I = S.info, C = S.comps, out = [];
  for (let i = 0; i < S.n; i++) {
    if (I.mult[i] > 1 && C && I.comp_start[i] >= 0) {
      if (I.is_companion[i]) continue;                    // 系统的成员在主条目那里一起列
      for (let k = I.comp_start[i]; k < I.comp_start[i] + I.comp_n[i]; k++) {
        out.push({ entry: C.entry[k], sysEntry: i, id: C.id[k], spt: C.spt[k], mass: C.mass[k], teff: C.teff[k], MV: C.MV[k] === null ? 99 : C.MV[k],
          V: C.V[k] === null ? 99 : C.V[k], phase: C.phase[k], BV: C.BV[k], var: C.var[k], letter: C.letter[k], res: C.res[k] });
      }
    } else {
      out.push({ entry: i, sysEntry: i, id: I.id[i], spt: I.spt[i], mass: I.mass[i], teff: I.teff[i], MV: I.MV[i], V: null,
        phase: I.phase[i], BV: S.raw[i * NF + 7], var: S.raw[i * NF + 11] | 0, letter: "", res: I.resolvable[i] });
    }
  }
  return (S._units = out);
}

function sptProps(sp) {
  const parts = (sp || "").split("+").filter(Boolean);
  const letters = [], lums = [];
  for (const x of parts) {
    if (x.startsWith("D")) { letters.push("D"); lums.push("D"); }
    else if (x !== "—") { letters.push(x[0]); lums.push(LUM_GROUP(x.replace(/^[A-Z][\d.]*/, ""))); }
  }
  return { letters, lums };
}

function starProps(i) {
  // 一颗星的可搜索字段：光谱型拆成字母和光度级。合成光谱（如 G7III+DA）两个成分都算
  const parts = (S.info.spt[i] || "").split("+").filter(Boolean);
  const letters = [], lums = [];
  for (const sp of parts) {
    if (sp.startsWith("D")) { letters.push("D"); lums.push("D"); }
    else { letters.push(sp[0]); lums.push(LUM_GROUP(sp.replace(/^[A-Z][\d.]*/, ""))); }
  }
  return { letters, lums };
}

function initSearch() {
  const $ = id => document.getElementById(id);
  $("qSpt").innerHTML = ["O", "B", "A", "F", "G", "K", "M", "D"].map(c =>
    `<label><input type="checkbox" value="${c}"> ${c === "D" ? "白矮星" : c}</label>`).join("");
  $("qPh").innerHTML = `<option value="">任意</option>` + Object.entries(PHASE_CN).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
  const vtypes = (S.info && S.info.var_types) || ["", ...Object.keys(VAR_CN)];
  $("qVar").innerHTML = `<option value="">任意</option><option value="none">非变星</option><option value="any">任意变星</option>` +
    vtypes.slice(1).map((k, j) => `<option value="${j + 1}">${VAR_CN[k] || k}</option>`).join("");
  const toggle = on => {
    const show = on === undefined ? $("search").style.display !== "block" : on;
    $("search").style.display = show ? "block" : "none";
    document.body.classList.toggle("searching", show);
  };
  $("bSearch").onclick = () => toggle();
  $("sClose").onclick = () => toggle(false);
  $("qGo").onclick = runSearch;
  $("qClear").onclick = () => {
    $("search").querySelectorAll("input[type=number], input[type=text]").forEach(x => x.value = "");
    $("search").querySelectorAll("select").forEach(x => x.selectedIndex = 0);
    $("qSpt").querySelectorAll("input").forEach(x => x.checked = false);
    S.found = []; $("sres").innerHTML = "";
  };
  $("search").addEventListener("keydown", e => { if (e.key === "Enter") runSearch(); });
  $("sres").onclick = e => {
    const tr = e.target.closest("tr[data-i]"); if (!tr) return;
    goToStar(+tr.dataset.i);
  };
  return toggle;
}

function runSearch() {
  const $ = id => document.getElementById(id);
  if (!S.info) { $("sres").textContent = "星表信息还在读取，稍等再试"; return; }
  const num = id => { const v = $(id).value.trim(); return v === "" ? null : +v; };
  const rng = (a, b) => [num(a), num(b)];
  const inR = (x, [lo, hi]) => (lo === null || x >= lo) && (hi === null || x <= hi);
  const q = {
    id: $("qId").value.trim().toLowerCase(), V: rng("qV0", "qV1"), d: rng("qD0", "qD1"),
    letters: [...$("qSpt").querySelectorAll("input:checked")].map(x => x.value),
    lum: $("qLum").value, ph: $("qPh").value, vr: $("qVar").value, mult: $("qMult").value,
    cl: $("qCl").value, clN: $("qClN").value.trim(), T: rng("qT0", "qT1"), M: rng("qM0", "qM1"),
    A: rng("qA0", "qA1"), MV: rng("qMV0", "qMV1"), E: rng("qE0", "qE1"), B: rng("qB0", "qB1"),
    up: $("qUp").checked, vis: $("qVis").checked,
  };
  const I = S.info, r = S.raw, st = skyState();
  const lim = S.mode === 0 ? S.meta.naked_eye_limit : S.meta.telescope_limit;
  const units = searchUnits();
  const out = [];
  for (const u of units) {
    const i = u.entry, s0 = u.sysEntry, o = i * NF;
    const shown = S.mag[i];                               // 天上画出来的亮度（分不开的系统是合成亮度）
    if (q.vis && !(shown <= lim)) continue;
    if (!inR(u.V === null ? shown : u.V, q.V) || !inR(I.dist_pc[s0], q.d)) continue;
    if (q.id && !u.id.toLowerCase().includes(q.id)) continue;
    if (q.letters.length || q.lum) {
      const p = sptProps(u.spt);
      if (q.letters.length && !p.letters.some(x => q.letters.includes(x))) continue;
      if (q.lum && !p.lums.includes(q.lum)) continue;
    }
    if (q.ph && u.phase !== q.ph) continue;
    if (q.vr) {
      const vt = u.var;
      if (q.vr === "none" ? vt !== 0 : q.vr === "any" ? vt === 0 : vt !== +q.vr) continue;
    }
    if (q.mult) {
      const m = I.mult[s0], rs = u.res;
      if (q.mult === "m" ? m < 2 : q.mult === "t" ? (u.letter === "A" || u.letter === "" || rs !== "telescope") :
          q.mult === "n" ? (u.letter === "A" || u.letter === "" || rs !== "naked_eye") : m !== +q.mult) continue;
    }
    if (q.cl === "y" && !I.cluster[s0]) continue;
    if (q.cl === "n" && I.cluster[s0]) continue;
    if (q.clN && !(I.cluster[s0] || "").includes(q.clN)) continue;
    if (!inR(u.teff, q.T) || !inR(u.mass, q.M) || !inR(I.age_myr[s0], q.A) || !inR(u.MV, q.MV) ||
        !inR(I.A_V[s0], q.E) || !inR(u.BV, q.B)) continue;
    if (q.up) {
      const uu = [r[o] + r[o + 3] * st.kyr, r[o + 1] + r[o + 4] * st.kyr, r[o + 2] + r[o + 5] * st.kyr];
      if (M3.ap(st.Mfe2h, uu)[2] <= 0) continue;
    }
    out.push(u);
  }
  const key = $("qSort").value, sgn = key.startsWith("-") ? -1 : 1, k = key.replace("-", "");
  const val = u => k === "V" ? (u.V === null ? S.mag[u.entry] : u.V) : k === "d" ? I.dist_pc[u.sysEntry] : k === "mass" ? u.mass :
    k === "teff" ? u.teff : k === "MV" ? u.MV : I.age_myr[u.sysEntry];
  out.sort((a, b) => sgn * (val(a) - val(b)));
  S.found = [...new Set(out.map(u => u.entry))];
  const SHOW = 300;
  const fmtD = d => d < 100 ? d.toFixed(1) : d.toFixed(0);
  $("sres").innerHTML = `<div class="dim">找到 ${out.length} 颗恒星${out.length > SHOW ? `，列出前 ${SHOW} 颗` : ""}；点一行转过去。星等是这颗星自己的亮度</div>` +
    `<table><tr class="dim"><td>编号</td><td>星等</td><td>光谱</td><td>距离 pc</td><td></td></tr>` +
    out.slice(0, SHOW).map(u => {
      const vt = u.var;
      const tag = [vt ? I.var_types[vt] : "", I.mult[u.sysEntry] > 1 ? `×${I.mult[u.sysEntry]}` : "", I.cluster[u.sysEntry] || ""].filter(Boolean).join(" ");
      const V = u.V === null ? S.mag[u.entry] : u.V;
      return `<tr data-i="${u.entry}"><td>${u.id.slice(0, 6)}…${u.id.slice(-4)}</td><td>${V >= 99 ? "—" : V.toFixed(2)}</td><td>${u.spt}</td>` +
        `<td>${fmtD(I.dist_pc[u.sysEntry])}</td><td class="dim">${tag}</td></tr>`;
    }).join("") + "</table>";
}

function goToStar(i) {
  const st = skyState(), r = S.raw, o = i * NF;
  const u = norm([r[o] + r[o + 3] * st.kyr, r[o + 1] + r[o + 4] * st.kyr, r[o + 2] + r[o + 5] * st.kyr]);
  const h = M3.ap(st.Mfe2h, u);
  const alt = Math.asin(h[2]) / D2R;
  S.az = (Math.atan2(h[0], h[1]) / D2R + 360) % 360;
  S.alt = S.see ? alt : Math.max(-10, alt);
  S.sel = i; showInfo(i, st);
  if (alt < 0 && !S.see) {
    document.getElementById("info").insertAdjacentHTML("afterbegin",
      `<div class="note" style="color:#e8a87a">此刻在地平线下 ${(-alt).toFixed(0)}°。可以打开"地面透视"，或改时刻等它升起。</div>`);
  }
}

load().catch(err => {
  document.getElementById("load").textContent = "读取失败：" + err.message + "（需要通过本地服务器打开，见 README）";
  console.error(err);
});
