# Research Direction

Future work should stay within zero-shot monocular long-video depth estimation from RGB alone. Acceptable directions include:
- better ways of processing videos longer than a model's temporal window;
- better temporal modelling inside the window;
- test-time ensembling or refinement that uses only the frames (for example flipped or time-reversed passes);
- keeping one depth scale over minutes-long videos;
- training or fine-tuning strategies on non-benchmark data within the model-size and data budget in `scoring/CONSTRAINTS.md`.

The visible train clips are for measuring such ideas, not for fitting them.

Out of scope:
- using camera poses, intrinsics or depth that the evaluator does not provide;
- larger pretrained depth models or paid external APIs;
- training on or looking up the Bonn RGB-D Dynamic / KITTI recordings;
- evaluating only on other data.
