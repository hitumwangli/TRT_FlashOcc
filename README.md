# TensorRT Implement of FlashOcc Writen In C++ With Cuda Acceleration

## Inference Speed
All time units are in milliseconds (ms), 
(Warning) Nearest interpolation is used by default.

||TRT-Engine|Postprocess|mean Total|mIou|config
|---|---|---|---|---|---|
|NVIDIA 3090 FP16|5.06|0.01|5.07|31.95|[M0:FO(BEVDetOCC)-R50](https://github.com/Yzichen/FlashOCC/blob/master/projects/configs/flashocc/flashocc-r50-M0.py)
|NVIDIA 3090 FP16|6.55|0.01|6.56|32.08|[M1:FO(BEVDetOCC)-R50](https://github.com/Yzichen/FlashOCC/blob/master/projects/configs/flashocc/flashocc-r50.py)

## DataSet
The Project provides a test sample that can also be used for inference on the nuScenes dataset. When testing on the nuScenes dataset, you need to use the data_infos folder provided by this project. The data folder should have the following structure:

    └── data
        ├── nuscenes 
            ├── data_infos [Download1]
                ├── samples_infos
                    ├── sample0000.yaml
                    ├── sample0001.yaml
                    ├── ...
                ├── samples_info.yaml
                ├── time_sequence.yaml
            ├── samples
            ├── sweeps
            ├── ...
    └── debug_file_from_torch [Download2]
        ├── params_for_lss
        ├── torch_out_for_c_debug_0
        ├── torch_out_for_c_debug_100
        └── torch_out_for_c_debug_sdfdsf
    └── onnx_for_c_trt [Download3]
        ├── flashocc-r50
        └── flashocc-r50-M0

- [Download1] can be downloaded from [Google drive](https://drive.google.com/file/d/1RkjzvDJH4ZapYpeGZerQ6YZyervgE1UK/view?usp=drive_link) or [Baidu Netdisk](https://pan.baidu.com/s/1TyPoP6OPbkvD9xDRE36qxw?pwd=pa1v)
- [Download2] can be downloaded from [Google drive](https://drive.google.com/file/d/1MxcApSI-CZMinuSOriJZ8JPIM_r-rmoX/view?usp=sharing)
- [Download3] can be downloaded from [Google drive](https://drive.google.com/file/d/1dlN27GDuTB-RcnLS87s-EEmcLpNz2dDS/view?usp=sharing)

## Environment
For desktop or server：

- CUDA 11.8
- cuDNN 8.6.0
- TensorRT 8.4.0.6
- yaml-cpp
- Eigen3
- libjpeg

For Jetson AGX Orin [no check]

- Jetpack 5.1.1
- CUDA 11.4.315
- cuDNN 8.6.0
- TensorRT 8.4.0.6
- yaml-cpp
- Eigen3
- libjpeg
  
## Compile && Run

### 1. Export onnx

- You can direct use the onnx from the [Download3](https://drive.google.com/file/d/1dlN27GDuTB-RcnLS87s-EEmcLpNz2dDS/view?usp=sharing) above.
- or export onnx follow [Quick Test Via TensorRT In MMDeploy](https://github.com/Yzichen/FlashOCC/blob/master/doc/mmdeploy_test.md)
```shell
python tools/analysis_tools/benchmark_trt.py $config $engine
```

### 2. Build
```shell
rm -r ./build/*
cd ./build
clear
cmake .. && make
```

### 3. Quick validation of accuracy deviation between torch and trt via single-frame testing
#### 3.1 export trt
```shell
rm ../model/bevdet_fp16.engine
./export ../onnx_for_c_trt/flashocc-r50/bevdet_fp16_fuse_for_c_and_trt.onnx ../model/bevdet_fp16.engine
```

#### 3.2 inference with input from torch
```shell
./build/bevdemo ./flashocc_quick_check_for_alignation_with_torch.yaml
```
#### 3.3 check validation of accuracy deviation
```python
torch_cls_occ_label = np.loadtxt("./debug_file_from_torch/torch_out_for_c_debug_0/torch_cls_occ_label.txt")
c_cls_occ_label = np.loadtxt("./c++_output/bevdet_occ_label_result_cpu.txt")
print('acc is: ', (torch_cls_occ_label == c_cls_occ_label).sum()/(c_cls_occ_label.shape[0]))
# acc is: 0.998534375
```


### 4. Inference
#### 4.1 export trt
```shell
rm ../model/bevdet_fp16.engine
./export ../onnx_for_c_trt/flashocc-r50-M0/bevdet_fp16_fuse_for_c_and_trt.onnx ../model/bevdet_fp16.engine
```

#### 4.2 inference
```shell
./build/bevdemo ./flashocc.yaml
```



## References
- [FlashOcc](https://github.com/Yzichen/FlashOCC)
- [Mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [NuScenes](https://www.nuscenes.org/)
- [BEVDet_TRT](https://github.com/LCH1238/bevdet-tensorrt-cpp)
