# Problem Definition

## Research Question

Estimate dense depth for long monocular videos so that the predictions are both spatially accurate and temporally consistent over hundreds to thousands of frames. The input is an RGB video only (no camera poses, no intrinsics, no depth). The output is one relative depth map per frame. Relative means affine-invariant in inverse depth, but one scale and shift must hold for the whole video: if the depth scale of frame 1400 does not match frame 10, the prediction is wrong.

## Why It Matters

Video depth feeds robotics, mapping, 3D video conversion and video editing. Single-image depth estimators give plausible per-frame maps, but on video they flicker. Any model that sees only a bounded number of frames at a time tends to produce scale jumps between the pieces it processed, or a scale that drifts slowly over a long video. Downstream uses such as fusing frames into a point cloud need one consistent geometry for the whole clip.

## Setting in this environment

The videos are real recordings from two public RGB-D / LiDAR video benchmarks that the paper evaluates on, used zero-shot:

- **Indoor, dynamic.** Bonn RGB-D Dynamic: a handheld Kinect-style camera moves through a lab while people walk, carry boxes or play with balloons. 640x480 at 30 fps. Ground truth is the sensor's registered depth (dense, with holes), and the camera motion comes from a motion-capture system.
- **Outdoor, driving.** KITTI: the left colour camera of a car driving through city, residential and road scenes. About 1242x375 at 10 fps. Ground truth is the official annotated LiDAR depth (sparse, mostly the lower part of the image), and the camera motion comes from the car's GPS/IMU.

Three settings are evaluated: 500-frame clips of both benchmarks (the paper's long-video protocol), and 1500-frame KITTI clips (2.5 minutes of driving, roughly 70 times longer than a 32-frame processing window, and the scene at the end shares nothing with the start).

## Success Criteria

A successful system:

- has low depth error after one global scale/shift alignment per video (AbsRel, delta1);
- is temporally consistent: depth reprojected from one frame to the next with the true camera motion agrees with the next frame's depth (Temporal Alignment Error);
- keeps one depth scale over the whole video, without jumps or slow drift;
- does this zero-shot, at the small-model scale and training-data budget given in `scoring/CONSTRAINTS.md`, so gains come from better video-depth methods rather than a much larger model, extra supervision or fitting the benchmarks themselves.
