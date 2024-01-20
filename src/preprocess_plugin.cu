/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "preprocess_plugin.h"
#include <map>
#include <cstring>
#include "common.h"

// kernel for GPU
template<typename T>
__global__ void preprocess_kernel(const uint8_t * src_dev, 
                                T* dst_dev, 
                                int src_row_step, 
                                int dst_row_step, 
                                int src_img_step, 
                                int dst_img_step,
                                int src_h, 
                                int src_w, 
                                float radio_h, 
                                float radio_w, 
                                float offset_h, 
                                float offset_w, 
                                const float * mean, 
                                const float * std,
                                int dst_h,
                                int dst_w,
                                int n){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= dst_h * dst_w * n) return;
    
    int i = (idx / n) / dst_w;
    int j = (idx / n) % dst_w;
    int k = idx % n;

	int pX = (int) roundf((i / radio_h) + offset_h);
	int pY = (int) roundf((j / radio_w) + offset_w);
 
	if (pX < src_h && pX >= 0 && pY < src_w && pY >= 0){
        int s1 = k * src_img_step + 0 * src_img_step / 3 + pX * src_row_step + pY;
        int s2 = k * src_img_step + 1 * src_img_step / 3 + pX * src_row_step + pY;
        int s3 = k * src_img_step + 2 * src_img_step / 3 + pX * src_row_step + pY;

        int d1 = k * dst_img_step + 0 * dst_img_step / 3 + i * dst_row_step + j;
        int d2 = k * dst_img_step + 1 * dst_img_step / 3 + i * dst_row_step + j;
        int d3 = k * dst_img_step + 2 * dst_img_step / 3 + i * dst_row_step + j;

		dst_dev[d1] = (static_cast<T>(src_dev[s1]) - static_cast<T>(mean[0])) / static_cast<T>(std[0]);
		dst_dev[d2] = (static_cast<T>(src_dev[s2]) - static_cast<T>(mean[1])) / static_cast<T>(std[1]);
		dst_dev[d3] = (static_cast<T>(src_dev[s3]) - static_cast<T>(mean[2])) / static_cast<T>(std[2]);
	}
}

void customPreprocessPlugin(const int n_img, const int src_img_h, const int src_img_w, 
    const int dst_img_h, const int dst_img_w, const int type_int,
    const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream,
    const float offset_h, const float offset_w, const float resize_radio){

    int src_row_step = src_img_w;
    int dst_row_step = dst_img_w;

    int src_img_step = src_img_w * src_img_h * 3;
    int dst_img_step = dst_img_w * dst_img_h * 3;

    dim3 grid(DIVUP(dst_img_h * dst_img_w * n_img,  NUM_THREADS));
    dim3 block(NUM_THREADS);

    int DataType_kFLOAT_int = int(nvinfer1::DataType::kFLOAT);
    int DataType_kHALF_int = int(nvinfer1::DataType::kHALF);

    if (type_int == DataType_kFLOAT_int)
    {
        preprocess_kernel<<<grid, block, 0, stream>>>(
                                                reinterpret_cast<const uint8_t *>(inputs[0]),
                                                reinterpret_cast<float *>(outputs[0]),
                                                src_row_step, 
                                                dst_row_step, 
                                                src_img_step,
                                                dst_img_step, 
                                                src_img_h, 
                                                src_img_w, 
                                                resize_radio,
                                                resize_radio, 
                                                offset_h, 
                                                offset_w, 
                                                reinterpret_cast<const float *>(inputs[1]), 
                                                reinterpret_cast<const float *>(inputs[2]),
                                                dst_img_h, 
                                                dst_img_w,
                                                n_img);
    }
    else if (type_int == DataType_kHALF_int)
    {
        // printf("pre : half\n");
        preprocess_kernel<<<grid, block, 0, stream>>>(
                                                reinterpret_cast<const uint8_t *>(inputs[0]),
                                                reinterpret_cast<__half *>(outputs[0]),
                                                src_row_step, 
                                                dst_row_step, 
                                                src_img_step,
                                                dst_img_step, 
                                                src_img_h, 
                                                src_img_w, 
                                                resize_radio,
                                                resize_radio, 
                                                offset_h, 
                                                offset_w, 
                                                reinterpret_cast<const float *>(inputs[1]), 
                                                reinterpret_cast<const float *>(inputs[2]),
                                                dst_img_h, 
                                                dst_img_w,
                                                n_img);

    }
    else
    {
        printf("\tUnsupport datatype!\n");
    }
        



    // switch (type_int)
    // {
    // // case static_cast<const int>(DataType_kFLOAT_int):
    // case reinterpret_cast<const int>(DataType_kFLOAT_int):
    //     // printf("pre : float\n");
    //     preprocess_kernel<<<grid, block, 0, stream>>>(
    //                                             reinterpret_cast<const uint8_t *>(inputs[0]),
    //                                             reinterpret_cast<float *>(outputs[0]),
    //                                             src_row_step, 
    //                                             dst_row_step, 
    //                                             src_img_step,
    //                                             dst_img_step, 
    //                                             src_img_h, 
    //                                             src_img_w, 
    //                                             resize_radio,
    //                                             resize_radio, 
    //                                             offset_h, 
    //                                             offset_w, 
    //                                             reinterpret_cast<const float *>(inputs[1]), 
    //                                             reinterpret_cast<const float *>(inputs[2]),
    //                                             dst_img_h, 
    //                                             dst_img_w,
    //                                             n_img);
    //     break;
    // // case static_cast<const int>(DataType_kHALF_int):
    // case reinterpret_cast<const int>(DataType_kHALF_int):
    //     // printf("pre : half\n");
    //     preprocess_kernel<<<grid, block, 0, stream>>>(
    //                                             reinterpret_cast<const uint8_t *>(inputs[0]),
    //                                             reinterpret_cast<__half *>(outputs[0]),
    //                                             src_row_step, 
    //                                             dst_row_step, 
    //                                             src_img_step,
    //                                             dst_img_step, 
    //                                             src_img_h, 
    //                                             src_img_w, 
    //                                             resize_radio,
    //                                             resize_radio, 
    //                                             offset_h, 
    //                                             offset_w, 
    //                                             reinterpret_cast<const float *>(inputs[1]), 
    //                                             reinterpret_cast<const float *>(inputs[2]),
    //                                             dst_img_h, 
    //                                             dst_img_w,
    //                                             n_img);

    //     break;
    // default: // should NOT be here
    //     printf("\tUnsupport datatype!\n");
    // }
        
}