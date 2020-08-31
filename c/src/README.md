# NNStreamer APIs

## Files
- nnstreamer-capi-pipeline.c - API to make pipeline with NNStreamer
- nnstreamer-capi-single.c - API to run a single model with NNStreamer, independent of GStreamer
- nnstreamer-capi-util.c - Utility functions for CAPI

## Comparison of Single API

### Latency & Running-time
Below shows the comparison of old pipeline-based vs Gst-less Single API - showing reduction in latency and running-time for the API (tested with tensorflow-lite).

These values averaged over 10 continuous runs.

|  | Open (us)      | Invoke (ms)           | Close (us)  |
| --- |:-------------:|:-------------:|:-----:|
| New (cache warmup) |  658   | 5195 | 206 |
| Old (cache warmup) |  1228 | 5199   | 5245 |
| New (no warmup) | 1653 | 5205  | 201 |
| Old (no warmup) | 7201 | 5225  | 5299  |


These values are just for the first run.

| | Open (us)      | Invoke (ms)           | Close (us)  |
| --- |:-------------:|:-------------:|:-----:|
| New  |  12326   | 5231 | 2347 |
| Old |  58772 | 5250   | 52611 |

### Memory consumption

Comparison of the maximum 