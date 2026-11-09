# Method

The paper, "Video Depth Anything: Consistent Depth Estimation for Super-Long Videos" (arXiv 2501.12375), turns the Depth Anything V2 image model into a video depth model:

- **Architecture.** The frozen Depth Anything V2 ViT encoder processes every frame; the DPT head becomes a spatio-temporal head by inserting four temporal self-attention layers (attention along the time axis at each spatial position, absolute temporal position embeddings). The model consumes 32-frame clips.
- **Training.** Scale-shift-invariant spatial loss plus a temporal gradient matching loss (the change of depth between consecutive frames should match the ground truth change, on stable regions), on 0.55M synthetic video frames, jointly with distillation on unlabeled images. Only the head is trained.
- **Long-video inference (OI+KR).** The video is processed as 32-frame clips that advance by 22 frames. Each new clip contains 22 new frames, 8 overlap frames from the previous clip and 2 key frames (the first frame of the video and one earlier frame), so the network sees context it has already predicted. The new clip's predictions are scale/shift-aligned to the stored predictions of the key frames, and the 8 overlap frames are linearly cross-faded between the two clips. The paper's ablation (Table 5) shows this beats independent windows, overlap alignment and overlap interpolation alone.

## Implementation in this workspace

- `method/official/`: the released Video Depth Anything code (`video_depth_anything/video_depth.py::infer_video_depth` implements OI+KR with `INFER_LEN=32`, `OVERLAP=10`, `KEYFRAMES=[0,12,...]`, `INTERP_LEN=8`).
- `method/predict_vda.py`: predictor entry point (contract in `briefing/evaluation.md`); `--mode oi_kr` is the paper's inference, `--mode independent_windows` the no-stabilisation baseline. Both use the released VDA-S checkpoint (`data/downloads/video_depth_anything_vits.pth`, 28.4M parameters), 518-pixel input and fp32.
- `method/predictors/vda_oi_kr.json`, `baseline/predictors/vda_independent_windows.json`: the registered reference and baseline.
- `method/run_vda.py`: model loading and the independent-window loop (also a small CLI for PNG frame folders).

The checkpoint is used as released; nothing is retrained here.
