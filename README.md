# Video Depth Anything: Consistent Depth Estimation for Super-Long Videos

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2501.12375"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2501.12375-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2501.12375](https://arxiv.org/abs/2501.12375) |

## Summary

This workspace reproduces the core long-video inference claim of Video Depth Anything. It packages the official VDA-S implementation, a reproducible 128-frame RGB-D video benchmark, an independent-window baseline, and an artifact-only evaluator that measures spatial depth accuracy, temporal consistency, parameter count, and video training-data scale.

## Problem

Estimate dense metric-consistent depth for long monocular videos while keeping predictions spatially accurate and temporally stable across hundreds or thousands of frames. The benchmark asks for one depth map per input RGB frame.

## Run it

```bash
pip install git+https://github.com/zlab-princeton/scientist_world_cli
sw research 2501.12375 --mode innovate --agent claude-code
sw evaluate 2501.12375
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

