# Problem Definition

## Research Question

Estimate dense metric-consistent depth for long monocular videos while keeping predictions spatially accurate and temporally stable across hundreds or thousands of frames. The benchmark asks for one depth map per input RGB frame.

## Why It Matters

Long-video depth is useful for robotics, mapping, editing, and scene reconstruction. Image-only depth estimators often produce plausible single-frame maps but drift in scale or flicker over time, which makes the output hard to use in downstream video pipelines.

## Success Criteria

A successful system should:

- Predict per-frame depth with low relative error after the standard global affine alignment used by monocular depth benchmarks.
- Preserve temporal consistency, especially across long clips where independent short-window processing can introduce scale jumps.
- Maintain reasonable resource constraints, including model size and training-data scale, so gains are not obtained only by using a substantially larger model or more supervision.
