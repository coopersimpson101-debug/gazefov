/**
 * gazefov render — public API.
 *
 *   import { FoveatedSplatRenderer, computeLayout, FoveatedFrameDriver } from 'gazefov-render';
 */
export { FoveatedSplatRenderer } from './foveation/renderer';
export type { FoveatedTarget, FoveatedTimings, FoveatedView } from './foveation/renderer';
export { computeLayout, radiusPx, shadedPixelFraction } from './foveation/layout';
export type { FoveationConfig, FoveationLayout, Gaze } from './foveation/layout';
export { MAR0_DEG, MAR_SLOPE_AGGRESSIVE, MAR_SLOPE_CONSERVATIVE, centralPpd, fovealRadiusDeg, marDeg, minEccentricityForPitch } from './foveation/acuity';
export { luma, regionalPsnr, regionalSsim } from './foveation/metrics';
export { PACKET_SIZE, decodeGaze, encodeGaze, crc32 } from './gaze/protocol';
export type { GazePacket, EyeSample } from './gaze/protocol';
export { GazeFilter, OneEuro } from './gaze/filter';
export { PointerGazeSource, WebSocketGazeSource } from './gaze/source';
export type { GazeSample, GazeSource } from './gaze/source';
export { DEFAULT_POLICY, FoveatedFrameDriver } from './vanta/adapter';
export type { FoveationHost, FoveationPolicy, HostEyeView } from './vanta/adapter';
export { generateDemoScene } from './vendor/splatweb/io/procedural';
export type { SplatData } from './vendor/splatweb/io/splatData';
