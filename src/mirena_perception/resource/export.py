from ultralytics import YOLO
model = YOLO("/home/mirena/mirena_ws/src/mirena_perception/resource/best.pt")
model.export(
    format="engine",
    imgsz=640,          # 1. Explicitly lock the input resolution
    dynamic=False,      # 2. Disable dynamic axes (CRITICAL for speed)
    int8=False,         # 4. Turn off INT8 unless you have calibration data (see note below)
    workspace=4,        # 5. Give the TRT builder 4GB of RAM to find the fastest kernels
    batch=1,            # 6. Lock batch size to exactly 1
    simplify=True,      # 7. Optimize the ONNX graph structure before TRT conversion
    device=0            # 8. Force the builder to use the Jetson's GPU
)