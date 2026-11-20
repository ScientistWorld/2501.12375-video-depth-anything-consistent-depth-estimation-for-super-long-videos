# Train/Test Evaluation Design Notes (iteration 12)

- **Data.** Real long videos from two of the paper's benchmarks, used zero-shot:
  - Bonn RGB-D Dynamic: indoor scenes with moving people, 640x480 at 30 fps, registered Kinect depth, motion-capture camera poses.
  - KITTI: outdoor driving, left colour camera, about 1242x375 at 10 fps, official annotated LiDAR depth, GPS/IMU poses.
  - `data/fetch_clips.py` reads only the needed archive members with HTTP range requests and CRC-checks them.
  - Each Bonn RGB frame is paired with the depth map closest in time (within 20 ms). Pairing by list position drifts by up to 0.2 s because some depth frames were dropped.
- **Experiments.** `bonn_500` (first 500 frames of a sequence), `kitti_500` (500 frames from frame 5, where annotated depth starts) and `kitti_1500` (1500-frame clips).
  - Train (`data/train_clips.json`): 6 / 4 / 2 videos.
  - Test (`eval/test/clips.json`): 12 / 10 / 5 videos.
  - Sequences and drives are disjoint between the splits.
- **Poses.**
  - Bonn: camera-to-world = T_ROS^-1 T_k T_ROS T_m, as on the dataset page. T_m's rotation is orthonormalised, and T_k is interpolated to the RGB timestamp.
  - KITTI: OXTS Mercator pose times the inverse of T_cam2_imu from the calibration files, as in pykitti.
  - Both were checked on ground-truth depth: with the true pose, the reprojection AbsRel median was 1.5% (Bonn) and 0.3% (KITTI), against 6.7% and 1.7% with no motion.
- **Predictor contract.**
  - Manifests in `method/predictors/` and `baseline/predictors/` hold a `command`. Each evaluator runs `command --jobs jobs.json` once per predictor.
  - Inputs are native-resolution RGB PNG frames under opaque ids.
  - The output is `prediction [T, ceil(H/2), ceil(W/2)]`: inverse depth on every 2nd row/column. This keeps prediction files at about 7 GB per predictor for the test set. Subsampling pixels uniformly leaves the means of the metrics unchanged.
- **Metrics.** These follow the official VDA benchmark: one least-squares scale/shift per video in inverse depth (prediction clipped at 1e-3), depth clipped to max_depth (10 m Bonn, 80 m KITTI), and per-frame AbsRel and delta1 averaged over frames.
  - TAE follows the paper's Eq. 5 (x100) on all pixels, with the true relative poses. Unlike the official script, it uses a z-buffer: the closest surface wins.
  - scale_drift is the std over frames of the per-frame median log ratio (x100).
- **Gate.** A run fails if:
  - any output is missing, misshaped or non-finite, or the predictor errors;
  - or in any experiment it does not beat the static per-pixel template (fitted to the test labels) on absrel and delta1, or it reaches TAE 30 or drift 50.

  A failed run gets the gate values on every metric, and the reason goes under `_status`.
- **Hidden-test isolation.** The predictor runs under `unshare` in new user, mount and PID namespaces:
  - `eval/test/` is covered by an empty tmpfs, and the input frame directory is bind-mounted read-only. Both mounts are locked by a nested user namespace.
  - `/proc` is private.
  - Ground truth is read only after the predictor returns.
  - The evaluator self-tests this at start-up. If the platform does not allow it, the predictor runs without isolation on copied (not hard-linked) frames, and the detail file records why.
- **Outputs.** Score files are always rewritten from scratch. Neither evaluator reads the other's score files, sidecars, or anything the method wrote apart from `prediction`. No wall-clock limit is applied by default.

Reasons for each choice and the audit numbers are in `HARDENING.md`.
