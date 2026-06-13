# Overview

This workspace reproduces the core long-video inference claim of Video Depth Anything. It packages the official VDA-S implementation, a reproducible RGB-D video benchmark, an independent-window baseline, and an artifact-only evaluator that measures spatial depth accuracy, temporal consistency, parameter count, and video training-data scale.

The first submitted job runs both inference strategies on the fixed synthetic long video and writes comparable metrics to `scoring/scores.json`.
