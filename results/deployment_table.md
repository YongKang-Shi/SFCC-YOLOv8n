| Model | Params (M) | GFLOPs | Weights (MB) | ONNX (MB) | Latency (ms) | FPS |
|---|---|---|---|---|---|---|
| YOLOv5nu | 2.514 | 7.21 | 10.23 | 10.29 | 15.8 | 63.3 |
| YOLO11n | 2.596 | 6.47 | 10.59 | 10.63 | 17.5 | 57.1 |
| YOLOv10n | 2.718 | 8.45 | 11.13 | 9.36 | 17.6 | 56.8 |
| YOLOv8n | 3.017 | 8.22 | 12.21 | 12.29 | 18.1 | 55.2 |
| **SFCC-YOLOv8n (this work)** | **2.594** | 7.40 | **10.56** | 10.64 | 18.7 | 53.5 |

Notes: all models were measured on the same machine with the same protocol - ONNX Runtime 1.23
(CPUExecutionProvider, Intel Core i7-14650HX), 640 x 640 input, batch = 1, fp32, median of three
repetitions after warm-up. Weight size is the fp32 state_dict; ONNX models were exported with
opset 12 and simplified with onnxslim. These are the values reported in Table 7 of the manuscript.
