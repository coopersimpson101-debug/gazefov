// gazefov — composite the foveal inset over the up-sampled periphery with a smooth radial blend.
//
//   w(r) = 1 - smoothstep(r_inner, r_outer, |p - gaze|)        (r in output pixels)
//   out  = mix(bilinear(periphery, p), inset(p), w)
//
// Both layers hold premultiplied colour (splatweb's blend convention), so the mix is linear and
// alpha stays consistent for the host's own compositing.

struct CompositeParams {
  gaze: vec2<f32>,        // output pixels, y down
  radii: vec2<f32>,       // r_inner, r_outer (px)
  inset_origin: vec2<f32>,// output pixels, y down (top-left)
  inset_size: vec2<f32>,
  out_size: vec2<f32>,
  debug: f32,             // > 0: tint the blend band (for figures only)
  _pad: f32,
};

@group(0) @binding(0) var<uniform> cp: CompositeParams;
@group(0) @binding(1) var periphery: texture_2d<f32>;
@group(0) @binding(2) var inset: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

struct VSOut { @builtin(position) position: vec4<f32> };

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  // Full-screen triangle.
  let p = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
  var o: VSOut;
  o.position = vec4<f32>(p * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
  return o;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let p = in.position.xy; // pixel centre, y down
  let per = textureSampleLevel(periphery, samp, p / cp.out_size, 0.0);
  let d = distance(p, cp.gaze);
  var w = 1.0 - smoothstep(cp.radii.x, cp.radii.y, d);
  let q = p - cp.inset_origin;
  let inside = all(q >= vec2<f32>(0.0)) && all(q < cp.inset_size);
  var c = per;
  if (inside && w > 0.0) {
    let fov = textureLoad(inset, vec2<i32>(floor(q)), 0);
    c = mix(per, fov, w);
  }
  if (cp.debug > 0.0 && w > 0.0 && w < 1.0) {
    c = vec4<f32>(mix(c.rgb, vec3<f32>(0.40, 0.91, 0.98) * c.a, 0.25), c.a);
  }
  return c;
}
