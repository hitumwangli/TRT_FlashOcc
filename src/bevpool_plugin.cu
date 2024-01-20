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

#include "bevpool_plugin.h"
#include "common.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

// input[0] depth            b*n x d x h x w
// input[1] feat             b*n x c x h x w
// input[2] ranks_depth      m
// input[3] ranks_feat       m
// input[4] ranks_bev        m
// input[5] interval_starts  k
// input[6] interval_lengths k

// out[0]   bevfeat          b x c x h x w    

template<typename T1, typename T2>
__global__ void bev_pool_v2_kernel(int channel, 
                                    int n_intervals,
                                    int map_size,
                                    int img_size,
                                    const T1 *__restrict__ depth,
                                    const T1 *__restrict__ feat,
                                    const int *__restrict__ ranks_depth,
                                    const int *__restrict__ ranks_feat,
                                    const int *__restrict__ ranks_bev,
                                    const int *__restrict__ interval_starts,
                                    const int *__restrict__ interval_lengths,
                                    T2 * __restrict__ out) {
    CUDA_1D_KERNEL_LOOP(idx, n_intervals * channel){
        int index = idx / channel;    // bev grid index
        int curr_c = idx % channel;    // channel index
        int interval_start = interval_starts[index];  
        int interval_length = interval_lengths[index];  

        int curr_step = curr_c * img_size;
        int chan_step = channel * img_size;

        T2 sum = 0;

        int feat_offset = 0;
        for(int i = 0; i < interval_length; i++){
            feat_offset = ranks_feat[interval_start + i] / img_size * chan_step + 
                          curr_step + ranks_feat[interval_start + i] % img_size;
  
            sum += static_cast<T2>(feat[feat_offset]) * static_cast<T2>(depth[ranks_depth[interval_start + i]]);
        }
        out[curr_c * map_size + ranks_bev[interval_start]] = sum;
    }
}

void customBEVPoolPlugin(
    const int channel, const int n_intervals, const int map_size, const int img_size,
    const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
    const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept {

    dim3 grid(GET_BLOCKS(n_intervals * channel));
    dim3 block(NUM_THREADS);

    // printf("BEVPool input depth %s\n", dataTypeToString(inputDesc[0].type).c_str());
    // printf("BEVPool input  feat %s\n", dataTypeToString(inputDesc[1].type).c_str());
    // printf("BEVPool output feat %s\n", dataTypeToString(outputDesc[0].type).c_str());

    switch (int(outputDesc[0].type))
    {
    case int(nvinfer1::DataType::kFLOAT):
        if(inputDesc[0].type == nvinfer1::DataType::kFLOAT){
            // printf("bevpool : fp32 fp32\n");
            bev_pool_v2_kernel<float, float><<<grid, block, 0, stream>>>(
                                                        channel, 
                                                        n_intervals,
                                                        map_size,
                                                        img_size,
                                                        reinterpret_cast<const float *>(inputs[0]),
                                                        reinterpret_cast<const float *>(inputs[1]),
                                                        reinterpret_cast<const int *>(inputs[2]),
                                                        reinterpret_cast<const int *>(inputs[3]),
                                                        reinterpret_cast<const int *>(inputs[4]),
                                                        reinterpret_cast<const int *>(inputs[5]),
                                                        reinterpret_cast<const int *>(inputs[6]),
                                                        reinterpret_cast<float *>(outputs[0]));
        }
        else{
            // printf("bevpool : fp16 fp32\n");
            bev_pool_v2_kernel<__half, float><<<grid, block, 0, stream>>>(
                                                        channel, 
                                                        n_intervals,
                                                        map_size,
                                                        img_size,
                                                        reinterpret_cast<const __half *>(inputs[0]),
                                                        reinterpret_cast<const __half *>(inputs[1]),
                                                        reinterpret_cast<const int *>(inputs[2]),
                                                        reinterpret_cast<const int *>(inputs[3]),
                                                        reinterpret_cast<const int *>(inputs[4]),
                                                        reinterpret_cast<const int *>(inputs[5]),
                                                        reinterpret_cast<const int *>(inputs[6]),
                                                        reinterpret_cast<float *>(outputs[0]));
        }
        break;
    case int(nvinfer1::DataType::kHALF):
        if(inputDesc[0].type == nvinfer1::DataType::kFLOAT){
            // printf("bevpool : fp32 fp16\n");
            bev_pool_v2_kernel<float, __half><<<grid, block, 0, stream>>>(
                                                        channel, 
                                                        n_intervals,
                                                        map_size,
                                                        img_size,
                                                        reinterpret_cast<const float *>(inputs[0]),
                                                        reinterpret_cast<const float *>(inputs[1]),
                                                        reinterpret_cast<const int *>(inputs[2]),
                                                        reinterpret_cast<const int *>(inputs[3]),
                                                        reinterpret_cast<const int *>(inputs[4]),
                                                        reinterpret_cast<const int *>(inputs[5]),
                                                        reinterpret_cast<const int *>(inputs[6]),
                                                        reinterpret_cast<__half *>(outputs[0]));
        }
        else{
            // printf("bevpool : fp16 fp16\n");
            bev_pool_v2_kernel<__half, __half><<<grid, block, 0, stream>>>(
                                                        channel, 
                                                        n_intervals,
                                                        map_size,
                                                        img_size,
                                                        reinterpret_cast<const __half *>(inputs[0]),
                                                        reinterpret_cast<const __half *>(inputs[1]),
                                                        reinterpret_cast<const int *>(inputs[2]),
                                                        reinterpret_cast<const int *>(inputs[3]),
                                                        reinterpret_cast<const int *>(inputs[4]),
                                                        reinterpret_cast<const int *>(inputs[5]),
                                                        reinterpret_cast<const int *>(inputs[6]),
                                                        reinterpret_cast<__half *>(outputs[0]));
        }
        break;
    default: // should NOT be here
        printf("\tUnsupport datatype!\n");
    }
}