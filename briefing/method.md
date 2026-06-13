# Method

The paper, "Video Depth Anything: Consistent Depth Estimation for Super-Long Videos", adapts the Depth Anything V2 monocular depth model to long videos.

The implemented method follows the released Video Depth Anything code:

- Start from a Depth Anything V2 image encoder and use a video depth head with temporal self-attention inserted into the DPT head.
- Process videos as fixed-length clips.
- During long-video inference, keep overlap frames between clips, use selected key frames as references for the next clip, and interpolate overlap regions to reduce scale jumps between windows.
- Use the released VDA-S checkpoint for this reproduction. The checkpoint corresponds to the paper's small model family with about 28.4M parameters.

The full paper trains on 0.55M video frames plus image datasets using a scale-shift-invariant spatial loss, a temporal gradient matching loss, and an image-model distillation loss. This environment focuses first on reproducing the core long-video inference claim with the released checkpoint and the official algorithm.
