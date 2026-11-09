# Experiments

Both experiments use hidden, procedurally rendered indoor corridor videos (static scenes, moving and turning camera, changing exposure, fog and noise) with exact depth and camera motion. Methods see only RGB frames. All metrics use one depth scale and shift per video, so every frame of a video has to agree on scale.

## long_video_500

Six 500-frame videos (about 40 seconds at 12 fps), the length of the paper's long-video evaluation. The camera alternately looks far down the corridor and faces nearby walls, so the visible depth range changes many times within a video. This establishes whether a video depth method stays accurate (AbsRel, delta1) and consistent from frame to frame (Temporal Alignment Error) while the scene content changes, and whether its depth scale stays the same across the whole clip (scale drift). The comparison point is the same network applied to fixed windows independently, which is markedly worse on all four metrics here.

## super_long_video_1500

Two 1500-frame videos (about two minutes). By the end of each video the camera has passed through several rooms and sees nothing that was visible at the start, so a method has to carry one consistent depth scale over a long horizon rather than anchoring to shared content. This establishes whether accuracy and consistency hold up as videos get much longer than any processing window, which is the setting where long-horizon scale drift would show up.
