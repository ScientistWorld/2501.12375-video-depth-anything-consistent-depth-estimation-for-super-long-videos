# Research Direction

Future work should stay within monocular long-video depth estimation from RGB alone. Acceptable directions include better ways of processing videos longer than a model's temporal window, better temporal modelling inside the window, test-time ensembling or refinement that uses only the frames, keeping one depth scale over minutes-long videos, and training or fine-tuning strategies within the model-size and data budget in `scoring/CONSTRAINTS.md`.

Out of scope: using camera poses, intrinsics or depth that the evaluator does not provide, larger pretrained depth models, paid external APIs, evaluating only on other data, or gaining performance by reproducing the rendered scenes' labels.
