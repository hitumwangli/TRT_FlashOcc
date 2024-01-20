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

#include "gatherbev_plugin.h"
#include "common.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
using namespace nvinfer1;

// input[0] == adj_feat   b x 8 x 80 x 128 x 128
// input[1] == curr_feat  b x 80 x 128 x 128
// input[2] == flag       b x 1
// out[0]                 b x (8+1)*80 x 128 x 128
template<typename T>
__global__ void copy_feat_kernel(int nthreads, // b * (adj_num + 1) * map_size
                                int adj_num,
                                int channel,
                                int map_size,
                                const T* adj_feats,
                                const T* curr_feat,
                                const int* flag,
                                T* out_feats){
    CUDA_1D_KERNEL_LOOP(idx, nthreads){
        int b = idx / ((adj_num + 1) * map_size);
        int n = (idx / map_size) % (adj_num + 1);
        int m = idx % map_size;

        int start = b * (adj_num + 1) * channel * map_size + n * channel * map_size + m;
        int end = start + channel * map_size;
        for(int i = start, c = 0; i < end; i += map_size, c++){
            if(flag[b] == 0 || n == 0){
                out_feats[i] = curr_feat[b * channel * map_size + c * map_size + m];
            }
            else{
                out_feats[i] = adj_feats[i - channel * map_size];
            }
        }
    }
}


void customGatherBEVPlugin(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc,
    const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept {

  // input[0] == adj_feat   b x 8 x 80 x 128 x 128
  // input[1] == curr_feat  b x 80 x 128 x 128
  // input[2] == flag       b x 1
  // out[0]                 b x (8+1)*80 x 128 x 128

    // int nthreads, // b * (adj_num + 1) * map_size
    // int adj_num,
    // int channel,
    // int map_size,

    // int flag = 0;
    // CHECK_CUDA(cudaMemcpy(&flag, inputs[2], sizeof(int), cudaMemcpyDeviceToHost));

    int b = inputDesc[0].dims.d[0];
    int adj_num = inputDesc[0].dims.d[1];
    int map_size = inputDesc[0].dims.d[3] * inputDesc[0].dims.d[4];
    int channel = inputDesc[0].dims.d[2];

    int feat_step = inputDesc[1].dims.d[1] * inputDesc[1].dims.d[2] * inputDesc[1].dims.d[3];

    int nthreads = b * (adj_num + 1) * map_size;

    dim3 grid(GET_BLOCKS(nthreads));
    dim3 block(NUM_THREADS);
    // printf("GatherBEV input adj_feats %s\n", dataTypeToString(inputDesc[0].type).c_str());
    // printf("GatherBEV input curr_feat %s\n", dataTypeToString(inputDesc[1].type).c_str());
    // printf("GatherBEV output bevfeats %s\n", dataTypeToString(outputDesc[0].type).c_str());

    switch (int(outputDesc[0].type))
    {
    case int(DataType::kFLOAT):
        // printf("gather : fp32\n");
        copy_feat_kernel<<<grid, block, 0, stream>>>(nthreads,
                                                    adj_num,
                                                    channel,
                                                    map_size,
                                                    reinterpret_cast<const float*>(inputs[0]),
                                                    reinterpret_cast<const float*>(inputs[1]),
                                                    reinterpret_cast<const int*>(inputs[2]),
                                                    reinterpret_cast<float*>(outputs[0]));
        

        break;
    case int(DataType::kHALF):
        // printf("gather : fp16\n");
        copy_feat_kernel<<<grid, block, 0, stream>>>(nthreads,
                                                    adj_num,
                                                    channel,
                                                    map_size,
                                                    reinterpret_cast<const __half*>(inputs[0]),
                                                    reinterpret_cast<const __half*>(inputs[1]),
                                                    reinterpret_cast<const int*>(inputs[2]),
                                                    reinterpret_cast<__half*>(outputs[0]));
        break;
    default: // should NOT be here
        printf("\tUnsupport datatype!\n");
    }
}
