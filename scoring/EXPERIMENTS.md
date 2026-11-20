# Experiments

All three experiments use hidden clips of real videos from public benchmarks the paper evaluates on, used zero-shot. Methods see only the RGB frames. All metrics use one depth scale and shift per video, so every frame of a video has to agree on scale. Ground truth is measured, not rendered: sensor depth with holes indoors, sparse LiDAR outdoors, and camera motion from motion capture or GPS/IMU.

## bonn_500

The first 500 frames (about 17 seconds at 30 fps) of 12 Bonn RGB-D Dynamic sequences. A handheld camera moves through a cluttered lab while people walk, carry boxes or play with balloons, so parts of every frame move independently of the camera. This follows the paper's long-video protocol (each video evaluated with up to 500 frames). It tests whether a video depth method:
- stays accurate (AbsRel, delta1) and consistent from frame to frame (Temporal Alignment Error) on a dynamic indoor scene;
- keeps one depth scale across the clip (scale drift).

## kitti_500

500-frame clips (50 seconds at 10 fps) of 10 KITTI drives, scored against the official annotated LiDAR depth up to 80 m. The scene streams past the camera, so the content of a processing window is replaced every few seconds. This is the paper's long-video protocol on an outdoor benchmark. It tests the same accuracy, consistency and scale questions over a much larger depth range.

## kitti_1500

1500-frame clips (2.5 minutes of driving) of 3 other KITTI drives, 5 clips in total. That is roughly 70 times a 32-frame processing window, and the end of each clip shares no content with its start. This is the paper's super-long-video setting, which it shows only qualitatively. It tests whether accuracy and a single consistent depth scale hold up when the video is far longer than anything a model sees at once, which is where slow scale drift and window-to-window jumps accumulate.
