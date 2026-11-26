// gazefov — splat rasterisation into one foveation layer.
//
// splatweb's preprocess pass runs ONCE per frame at the full virtual resolution and writes, per
// splat, a 2D Gaussian in full-resolution pixels (y up): centre + two 3σ half-axes. Every layer
// (low-res periphery, full-res foveal inset) re-uses that buffer and the single depth sort; this
// shader maps it into the layer with an affine pixel transform
//
//     p_layer = p_full * scale - offset
//
// (periphery: scale = s, offset = 0; inset: scale = 1, offset = inset origin).
//
// Down-scaling a Gaussian by s shrinks its covariance by s², so sub-pixel splats would alias in
// the low-res layer. We re-apply the anti-aliasing low-pass in *layer* pixels (as the EWA
// splatting filter of Zwicker et al. 2001 does in screen space) and keep the splat's integrated
// opacity constant by scaling α with sqrt(det Σ_scaled / det Σ_filtered) — the compensation used
// by Mip-Splatting's 2D filter (Yu et al. 2024). For scale = 1 the shader is identical to
// splatweb's render.wgsl.

struct Splat2D {
  center: vec2<f32>,
  axis1: vec2<f32>,
  axis2: vec2<f32>,
  color: vec2<u32>,
  depth: f32,
  _pad: f32,
};

struct LayerParams {
  viewport: vec2<f32>,  // layer size in pixels
  scale: vec2<f32>,     // full-res -> layer
  offset: vec2<f32>,    // subtracted after scaling (layer pixels)
  lowpass: f32,         // extra variance (layer px²) to top the filter back up; 0 for scale 1
  _pad: f32,
};

@group(0) @binding(0) var<uniform> lp: LayerParams;
@group(0) @binding(1) var<storage, read> splats2d: array<Splat2D>;
@group(0) @binding(2) var<storage, read> order: array<u32>;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec4<f32>,
};

const SIGMA_EXTENT: f32 = 3.0;

fn culled() -> VSOut {
  var out: VSOut;
  out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
  out.uv = vec2<f32>(0.0);
  out.color = vec4<f32>(0.0);
  return out;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  let s = splats2d[order[ii]];
  if (s.axis1.x == 0.0 && s.axis1.y == 0.0) { return culled(); }
  let center = s.center * lp.scale - lp.offset;
  var a1 = s.axis1 * lp.scale;
  var a2 = s.axis2 * lp.scale;
  let rg = unpack2x16float(s.color.x);
  let ba = unpack2x16float(s.color.y);
  var alpha = ba.y;

  if (lp.lowpass > 0.0) {
    // Σ = (a1 a1ᵀ + a2 a2ᵀ) / 9  (axes are 3σ), then Σ' = Σ + lowpass·I.
    let k = 1.0 / (SIGMA_EXTENT * SIGMA_EXTENT);
    let a = k * (a1.x * a1.x + a2.x * a2.x);
    let b = k * (a1.x * a1.y + a2.x * a2.y);
    let c = k * (a1.y * a1.y + a2.y * a2.y);
    let det0 = max(a * c - b * b, 1e-12);
    let af = a + lp.lowpass;
    let cf = c + lp.lowpass;
    let det1 = af * cf - b * b;
    alpha = alpha * sqrt(det0 / det1);
    let mid = 0.5 * (af + cf);
    let disc = sqrt(max(0.0, mid * mid - det1));
    let l1 = mid + disc;
    let l2 = max(mid - disc, 0.0);
    var v1 = select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), af >= cf);
    if (abs(b) >= 1e-9) { v1 = normalize(vec2<f32>(b, l1 - af)); }
    a1 = v1 * (SIGMA_EXTENT * sqrt(l1));
    a2 = vec2<f32>(-v1.y, v1.x) * (SIGMA_EXTENT * sqrt(l2));
  }

  // Quad entirely outside this layer: skip rasterisation work.
  let ext = abs(a1) + abs(a2);
  if (center.x + ext.x < 0.0 || center.x - ext.x > lp.viewport.x ||
      center.y + ext.y < 0.0 || center.y - ext.y > lp.viewport.y) { return culled(); }

  var out: VSOut;
  let corner = vec2<f32>(f32(vi & 1u), f32(vi >> 1u)) * 2.0 - vec2<f32>(1.0);
  let pix = center + corner.x * a1 + corner.y * a2;
  let ndc = pix / lp.viewport * 2.0 - vec2<f32>(1.0);
  out.position = vec4<f32>(ndc, s.depth, 1.0);
  out.uv = corner;
  out.color = vec4<f32>(rg, ba.x, alpha);
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  let alpha = min(0.99, in.color.a * exp(-4.5 * r2));
  if (alpha < 1.0 / 255.0) { discard; }
  return vec4<f32>(in.color.rgb * alpha, alpha);
}
