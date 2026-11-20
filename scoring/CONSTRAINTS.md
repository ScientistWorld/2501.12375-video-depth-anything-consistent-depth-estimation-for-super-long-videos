# Scoring Constraints

To keep scores comparable with the reference, a submitted method must:

- Take only the RGB frames handed to it by the evaluator (predictor contract in `briefing/evaluation.md`). It must not read ground truth, camera poses, intrinsics or calibration, the evaluator's code or files under `eval/`, or anything outside its own files and the given frames.
- Not identify the benchmark recordings in order to use their labels. This covers matching frames against a copy of Bonn RGB-D Dynamic or KITTI, and using their depth, poses or reconstructions. The test clips come from public benchmarks, so this is a rule checked on the submitted code, not something the evaluator can prevent.
- Be zero-shot with respect to the evaluation benchmarks, as in the paper ("all evaluation videos and images are excluded from the training datasets"):
  - Do not train, fine-tune or fit any parameter on Bonn RGB-D Dynamic or KITTI data, including the visible train clips.
  - The visible train clips are for evaluating and choosing between methods and inference settings. They are not supervision.
- Be inference-only at evaluation time and run offline. Train or fine-tune beforehand and load the weights.
- Stay at the small-model scale: all networks used at inference together have at most 30M parameters (the reference VDA-S has 28.4M). Larger pretrained depth models are out of scope, for example VDA-B/VDA-L, DepthCrafter, or Depth Anything V2 Base/Large.
- Use at most 0.55M video frames of depth supervision for any training or fine-tuning beyond the released checkpoint (the paper's video budget), and report the number used.
- Write one `prediction` array `[T, out_height, out_width]` of relative inverse depth per video on the evaluation grid. Do not change metric definitions, masks, alignment, the gate or the experiment set in `eval/`.

The benchmark rewards better long-video depth estimation, not larger backbones, extra supervision, external APIs, fitting the benchmark recordings, or access to the evaluation labels.
