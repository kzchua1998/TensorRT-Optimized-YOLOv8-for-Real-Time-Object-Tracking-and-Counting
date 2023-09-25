# TensorRT-Optimized-YOLOv8-for-Real-Time-Object-Tracking-and-Counting
`YOLOv8` using TensorRT accelerate for faster inference for object detection, tracking and intance segmentation.


# Results

- `object-detection`: ~88% `FPS`, ~390% `GPU-VRAM` improvement
- `instance-segmnetation`: ~55% `FPS`, ~253% `GPU-VRAM` improvement

| Models               | TensorRT Optimized               | FPS              | GPU-VRAM             |
|-- | :-: | :-: | :-: |
| **YOLOv8x-det + ByteTrack** | ✅ | **~32** | **~410MiB** |
| YOLOv8x-det + ByteTrack | ❌ | ~17 | ~1600MiB |
| **YOLOv8x-seg** | ✅ | **~28** | **~657MiB** |
| YOLOv8x-seg | ❌ | ~18 | ~1660MiB |

# Demo
### Vehicle Counting 
https://github.com/kzchua1998/TensorRT-Optimized-YOLOv8-for-Real-Time-Object-Tracking-and-Counting/assets/64066100/d69381b0-a4e2-48d7-a681-0eee06676639

### Human Tracking and Counting 
https://github.com/kzchua1998/TensorRT-Optimized-YOLOv8-for-Real-Time-Object-Tracking-and-Counting/assets/64066100/26feac1a-f8ea-452e-982b-b7bcb09a59f8


# Prepare the environment

1. Install `CUDA` follow [`CUDA official website`](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#download-the-nvidia-cuda-toolkit).

   🚀 RECOMMENDED `CUDA` >= 11.4

2. Install `TensorRT` follow [`TensorRT official website`](https://developer.nvidia.com/nvidia-tensorrt-8x-download).

   🚀 RECOMMENDED `TensorRT` >= 8.4

2. Install python requirements.

   ``` shell
   pip install -r requirements.txt
   ```

3. Install [`ultralytics`](https://github.com/ultralytics/ultralytics) package for ONNX export or TensorRT API building.

   ``` shell
   pip install ultralytics
   ```

5. Prepare your own PyTorch weight such as `yolov8s.pt` or `yolov8s-seg.pt`.




# Export End2End ONNX with NMS

You can export your YOLOv8 model weights from `ultralytics` with postprocess such as bbox decoder and `NMS` into ONNX model for both `detection` and `instance-segmentation` tasks.

``` shell
python export-det.py \
--weights yolov8s.pt \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--opset 11 \
--sim \
--input-shape 1 3 640 640 \
--device cuda:0
```

``` shell
python export-seg.py \
--weights yolov8s.pt \
--opset 11 \
--sim \
--input-shape 1 3 640 640 \
--device cuda:0
```

#### Description of all arguments

- `--weights` : The PyTorch model you trained.
- `--iou-thres` : IOU threshold for NMS plugin.
- `--conf-thres` : Confidence threshold for NMS plugin.
- `--topk` : Max number of detection bboxes.
- `--opset` : ONNX opset version, default is 11.
- `--sim` : Whether to simplify your onnx model.
- `--input-shape` : Input shape for you model, should be 4 dimensions.
- `--device` : The CUDA deivce you export engine .

You will get an onnx model whose prefix is the same as input weights.



## Build End2End Engine from ONNX

You can export TensorRT engine from ONNX by [`build.py` ](build.py).

Usage:

``` shell
python3 build.py \
--weights yolov8s.onnx \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--fp16  \
--device cuda:0
```

#### Description of all arguments

- `--weights` : The ONNX model you download.
- `--iou-thres` : IOU threshold for NMS plugin.
- `--conf-thres` : Confidence threshold for NMS plugin.
- `--topk` : Max number of detection bboxes.
- `--fp16` : Whether to export half-precision engine.
- `--device` : The CUDA deivce you export engine .

You can modify `iou-thres` `conf-thres` `topk` by yourself.


# Profile Your Engine

Profiling your engine enables you to identify and address performance bottlenecks, improve resource utilization, and tailor your model for specific deployment scenarios, ultimately leading to better inference performance and efficiency. If you want to profile the TensorRT engine:

Usage:

``` shell
python3 trt-profile.py --engine yolov8s.engine --device cuda:0
```

