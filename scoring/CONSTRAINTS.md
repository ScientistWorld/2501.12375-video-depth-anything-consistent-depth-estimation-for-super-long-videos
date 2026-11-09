# Scoring Constraints

To keep scores comparable with the reference, a submitted method must:

- Take only the RGB frames handed to it by the evaluator (predictor contract in `briefing/evaluation.md`). It must not read ground truth, camera poses, scene seeds, the evaluator's code or files under `eval/`, or anything outside its own files and the given frames, and it must not try to identify the rendered scene in order to reproduce its labels (for example by matching frames against the visible generator or by fitting the generator's scene parameters).
- Be inference-only at evaluation time and run offline. Train or fine-tune beforehand and load the weights.
- Stay at the small-model scale: all networks used at inference together have at most 30M parameters (the reference VDA-S has 28.4M). Larger pretrained depth models (for example VDA-L, DepthCrafter, Depth Anything V2 Large) are out of scope.
- Use at most 0.55M video frames of depth supervision for any training or fine-tuning beyond the released checkpoint (the paper's video budget), and report the number used. Videos rendered with `data/synthetic_rgbd.py` count towards it.
- Write one `prediction` array `[T, H, W]` of relative inverse depth per video. Do not change metric definitions, masks, alignment, the gate or the experiment set in `eval/`.

The benchmark rewards better long-video depth estimation, not larger backbones, extra supervision, external APIs, or access to the evaluation labels.
