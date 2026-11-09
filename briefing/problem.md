# Problem Definition

## Research Question

Estimate dense depth for long monocular videos so that the predictions are both spatially accurate and temporally consistent over hundreds to thousands of frames. The input is an RGB video only (no camera poses, no intrinsics, no depth); the output is one relative depth map per frame. Relative means affine-invariant in inverse depth, but a single scale and shift must hold for the whole video: if the depth scale of frame 1400 does not match frame 10, the prediction is wrong.

## Why It Matters

Video depth feeds robotics, mapping, 3D video conversion and video editing. Single-image depth estimators give plausible per-frame maps, but on video they flicker, and any model that sees only a bounded number of frames at a time tends to produce scale jumps between the pieces it processed, or a scale that drifts slowly over a long video. Downstream uses such as fusing frames into a point cloud need one consistent geometry for the whole clip.

## Setting in this environment

Videos are static indoor corridors (floor, ceiling, walls, partition walls with doorways, furniture) rendered with exact ray casting while a camera walks forward, sways, turns towards the walls and tilts. The visible depth range therefore changes a lot over time: some stretches look far down the corridor, others face a nearby wall. Lighting has a slow exposure drift, frame-to-frame gain flicker, fog and sensor noise. Frames are 144x192 at 12 fps. Two lengths are evaluated: 500 frames and 1500 frames (about two minutes, where the end of the video shares no content with the start).

## Success Criteria

A successful system:

- has low depth error after one global scale/shift alignment per video (AbsRel, delta1);
- is temporally consistent: depth reprojected from one frame to the next with the true camera motion agrees with the next frame's depth (Temporal Alignment Error);
- keeps one depth scale over the whole video, without jumps or slow drift;
- does this at the small-model scale and training-data budget given in `scoring/CONSTRAINTS.md`, so gains come from better video-depth methods rather than a much larger model or extra supervision.
