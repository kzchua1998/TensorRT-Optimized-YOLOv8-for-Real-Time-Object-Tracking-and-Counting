# YOLOv8-TensorRT

`YOLOv8` using TensorRT accelerate for faster inference


# Demo 
YOLOv8x-det TensorRT Engine + ByteTrack Inference with Torch `FPS: ~32`, `GPU-VRAM: ~410MiB`

| Models               | TensorRT Optimized               | FPS              | GPU-VRAM             |
| ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| ***YOLOv8x-det + ByteTrack*** | ***Yes*** | ***~32*** | ***~410MiB*** |
| YOLOv8x-det + ByteTrack | No | ~17 | ~1600MiB |
| ***YOLOv8x-seg*** | ***Yes*** | ***~28*** | ***~657MiB*** |
| YOLOv8x-seg | No | ~18 | ~1660MiB |
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

***NOTICE:***

Please use the latest `CUDA` and `TensorRT`, so that you can achieve the fastest speed !

If you have to use a lower version of `CUDA` and `TensorRT`, please read the relevant issues carefully !



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
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
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



# Build End2End Engine from ONNX using Python api

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



# Inference

## 1. Infer with python script

You can infer images with the engine by [`infer-det.py`](infer-det.py) .

Usage:

``` shell
python3 infer-det.py \
--engine yolov8s.engine \
--imgs data \
--show \
--out-dir outputs \
--device cuda:0
```

#### Description of all arguments

- `--engine` : The Engine you export.
- `--imgs` : The images path you want to detect.
- `--show` : Whether to show detection results.
- `--out-dir` : Where to save detection results images. It will not work when use `--show` flag.
- `--device` : The CUDA deivce you use.
- `--profile` : Profile the TensorRT engine.


# Profile you engine

If you want to profile the TensorRT engine:

Usage:

``` shell
python3 trt-profile.py --engine yolov8s.engine --device cuda:0
```

