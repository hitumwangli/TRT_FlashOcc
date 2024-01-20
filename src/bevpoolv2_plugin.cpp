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

#include "bevpoolv2_plugin.h"
#include "common.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

// /* customTRTBEVPoolV2的核函数接口部分 */
void bev_pool_v2(int c, int n_intervals, const float* depth, const float* feat,
    const int* ranks_depth, const int* ranks_feat, const int* ranks_bev,
    const int* interval_starts, const int* interval_lengths, float* out, cudaStream_t stream);
void bev_pool_v2_set_zero(int n_points, float* out);


namespace nvinfer1
{

/********************注册PluginCreator*****************************/
REGISTER_TENSORRT_PLUGIN(TRTBEVPoolV2Creator);


/*********************静态变量的申明*******************************/
// class TRTBEVPoolV2Creator
PluginFieldCollection    TRTBEVPoolV2Creator::fc_ {};
std::vector<PluginField> TRTBEVPoolV2Creator::attr_;


// class TRTBEVPoolV2
TRTBEVPoolV2::TRTBEVPoolV2(const std::string &name, int output_height, int output_width, int output_z):
    name_(name){
    m_.output_height = output_height;
    m_.output_width = output_width;
    m_.output_z = output_z;
}

TRTBEVPoolV2::TRTBEVPoolV2(const std::string &name, const void *buffer, size_t length):
    name_(name){
    memcpy(&m_, buffer, sizeof(m_));
}

TRTBEVPoolV2::~TRTBEVPoolV2(){
}

IPluginV2DynamicExt *TRTBEVPoolV2::clone() const noexcept {
    auto p = new TRTBEVPoolV2(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t TRTBEVPoolV2::getNbOutputs() const noexcept {
    return 1;
}
 
DataType TRTBEVPoolV2::getOutputDataType(int32_t index, DataType const *inputTypes, 
                                                                int32_t nbInputs) const noexcept {
    return DataType::kFLOAT;
}

DimsExprs TRTBEVPoolV2::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, 
                                        int32_t nbInputs, IExprBuilder &exprBuilder) noexcept {
    // input[0] == depth
    // input[1] == feat
    // input[2] == ranks_depth
    // input[3] == ranks_feat
    // input[4] == ranks_bev
    DimsExprs ret;
    // for slattened 2D bev feat
    ret.nbDims = 4;
    ret.d[0] = exprBuilder.constant(1); //Todo support batch>1
    ret.d[1] = exprBuilder.constant(m_.output_height);
    ret.d[2] = exprBuilder.constant(m_.output_width);
    ret.d[3] = inputs[1].d[3];
    // for 3D bev feat
    // ret.nbDims = 5;
    // ret.d[0] = exprBuilder.constant(1); //Todo support batch>1
    // ret.d[1] = inputs[1].d[3]; // exprBuilder.constant(32);
    // ret.d[2] = exprBuilder.constant(16);
    // ret.d[3] = exprBuilder.constant(mOutHeight);
    // ret.d[4] = exprBuilder.constant(mOutWidth);
    
    return ret; 
}

bool TRTBEVPoolV2::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut,
                                                    int32_t nbInputs, int32_t nbOutputs) noexcept {
    // // depth
    // if(pos == 0){
    //     return (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF) &&
    //              inOut[pos].format == TensorFormat::kLINEAR;
    // }
    // else if(pos == 1){ // feat
    //     return inOut[0].type == inOut[1].type && inOut[pos].format == TensorFormat::kLINEAR;
    // }
    // else if(pos == 7){ // out
    //     return (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF) &&
    //             inOut[pos].format == TensorFormat::kLINEAR;
    // }
    // else{
    //     return inOut[pos].type == DataType::kINT32 && inOut[pos].format == TensorFormat::kLINEAR;
    // }
    // return false;

    // input[0] == depth->kFLOAT
    // input[1] == feat->kFLOAT
    // input[2] == ranks_depth->kINT32
    // input[3] == ranks_feat->kINT32
    // input[4] == ranks_bev->kINT32
    // input[5] == interval_starts->kINT32
    // input[6] == interval_lengths->kINT32
    // output[0] == bev_feat->kFLOAT
    if (pos == 0 || pos==1 || pos == 7) {
        return (inOut[pos].type == DataType::kFLOAT &&
                inOut[pos].format == TensorFormat::kLINEAR);
    } else {
        return (inOut[pos].type == DataType::kINT32 &&
                inOut[pos].format == TensorFormat::kLINEAR);
    }

}

void TRTBEVPoolV2::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, 
                                    const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept {
    // ASSERT(nbInputs == 7);
    // ASSERT(nbOutputs == 1);
    return;
}

size_t TRTBEVPoolV2::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, 
                                const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept {
    return 0;
}

int TRTBEVPoolV2::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                            const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                            void *const *outputs, void *workSpace,
                            cudaStream_t stream) noexcept {
    nvinfer1::Dims feat_dims = inputDesc[1].dims; // bnhwc
    nvinfer1::Dims interval_dims = inputDesc[5].dims; // n
    nvinfer1::Dims out_dims = outputDesc[0].dims; //bhwc
    auto data_type = inputDesc[0].type;
    int num_points = out_dims.d[0]*out_dims.d[1]*out_dims.d[2]*out_dims.d[3];

    // int valid_feat_num = 302255;
    // int* ranks_depth_host = new int[valid_feat_num];
    // CHECK_CUDA(cudaMemcpy(ranks_depth_host, inputs[2], valid_feat_num * sizeof(int), cudaMemcpyDeviceToHost));
    // int* ranks_feat_host = new int[valid_feat_num];
    // int* ranks_bev_host = new int[valid_feat_num];
    // int unique_bev_num = 22092;
    // int* interval_starts_host_ptr = new int[unique_bev_num];
    // int* interval_lengths_host_ptr = new int[unique_bev_num];
    // int depth_feat_size = 6 * 88 * 16 * 44;
    // float* depth_feat_host_ptr = new float[depth_feat_size];
    // CHECK_CUDA(cudaMemcpy(depth_feat_host_ptr, inputs[0], depth_feat_size * sizeof(float), cudaMemcpyDeviceToHost));
    // int tran_feat_size = 6 * 16 * 44 * 64;
    // float* tran_feat_host_ptr = new float[tran_feat_size];
    // CHECK_CUDA(cudaMemcpy(tran_feat_host_ptr, inputs[1], tran_feat_size * sizeof(float), cudaMemcpyDeviceToHost));
    // for (int ind = 0; ind < 100; ind++) {
    //     std::cout << "ranks_depth_host[" << ind << "]:" << (int)ranks_depth_host[ind] << std::endl;
    // }
    // for (int ind = valid_feat_num-1-100; ind < valid_feat_num-1; ind++) {
    //     std::cout << "ranks_depth_host[" << ind << "]:" << (int)ranks_depth_host[ind] << std::endl;
    // }
    // for (int ind = 0; ind < 100; ind++) {
    //     std::cout << "depth_feat_host_ptr[" << ind << "]:" << (float)depth_feat_host_ptr[ind] << std::endl;
    // }
    // for (int ind = depth_feat_size-1-100; ind < depth_feat_size-1; ind++) {
    //     std::cout << "depth_feat_host_ptr[" << ind << "]:" << (float)depth_feat_host_ptr[ind] << std::endl;
    // }
    // for (int ind = 0; ind < 100; ind++) {
    //     std::cout << "tran_feat_host_ptr[" << ind << "]:" << (float)tran_feat_host_ptr[ind] << std::endl;
    // }
    // for (int ind = tran_feat_size-1-100; ind < tran_feat_size-1; ind++) {
    //     std::cout << "tran_feat_host_ptr[" << ind << "]:" << (float)tran_feat_host_ptr[ind] << std::endl;
    // }

    switch (data_type) {
        case nvinfer1::DataType::kFLOAT:
            bev_pool_v2_set_zero(num_points, (float *)outputs[0]);
            bev_pool_v2(feat_dims.d[3], interval_dims.d[0], (float *)inputs[0], (float *)inputs[1],
            (int *)inputs[2], (int *)inputs[3], (int *)inputs[4], (int *)inputs[5],(int *)inputs[6], (float *)outputs[0],
            stream);
            break;
        default:
            return 1;
            break;
    }

  return 0;
}

void TRTBEVPoolV2::destroy() noexcept {
    delete this;
    return;
}

int32_t TRTBEVPoolV2::initialize() noexcept {
    return 0;
}

void TRTBEVPoolV2::terminate() noexcept {
    return;
}

size_t TRTBEVPoolV2::getSerializationSize() const noexcept {
    return sizeof(m_);
}

void TRTBEVPoolV2::serialize(void *buffer) const noexcept {
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void TRTBEVPoolV2::setPluginNamespace(const char *pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
    return;
}

const char *TRTBEVPoolV2::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

const char *TRTBEVPoolV2::getPluginType() const noexcept {
    return TRTBEVPoolV2_PLUGIN_NAME;
}

const char *TRTBEVPoolV2::getPluginVersion() const noexcept {
    return TRTBEVPoolV2_PLUGIN_VERSION;
}

void TRTBEVPoolV2::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, 
                                                        IGpuAllocator *gpuAllocator) noexcept {
    return;
}

void TRTBEVPoolV2::detachFromContext() noexcept {
    return;
}

TRTBEVPoolV2Creator::TRTBEVPoolV2Creator() {
    attr_.clear();
    attr_.emplace_back(PluginField("output_height", nullptr, PluginFieldType::kINT32, 1));
    attr_.emplace_back(PluginField("output_width", nullptr, PluginFieldType::kINT32, 1));
    attr_.emplace_back(PluginField("output_z", nullptr, PluginFieldType::kINT32, 1));


    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

TRTBEVPoolV2Creator::~TRTBEVPoolV2Creator() {
}


IPluginV2DynamicExt *TRTBEVPoolV2Creator::createPlugin(const char *name, 
                                    const PluginFieldCollection *fc) noexcept {
    const PluginField *fields = fc->fields;

    int output_height = -1;
    int output_width = -1;
    int output_z = -1;

    for (int i = 0; i < fc->nbFields; ++i){
        if(std::string(fc->fields[i].name) == std::string("output_height")){
            output_height = *reinterpret_cast<const int *>(fc->fields[i].data);
        }
        else if(std::string(fc->fields[i].name) == std::string("output_width")){
            output_width = *reinterpret_cast<const int *>(fc->fields[i].data);
        }
        else if(std::string(fc->fields[i].name) == std::string("output_z")){
            output_z = *reinterpret_cast<const int *>(fc->fields[i].data);
        }
    }
    TRTBEVPoolV2 *pObj = new TRTBEVPoolV2(name, output_height, output_width, output_z);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

IPluginV2DynamicExt *TRTBEVPoolV2Creator::deserializePlugin(const char *name, 
                                        const void *serialData, size_t serialLength) noexcept {
    TRTBEVPoolV2 *pObj = new TRTBEVPoolV2(name, serialData, serialLength);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

void TRTBEVPoolV2Creator::setPluginNamespace(const char *pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
    return;
}

const char *TRTBEVPoolV2Creator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

const char *TRTBEVPoolV2Creator::getPluginName() const noexcept {
    return TRTBEVPoolV2_PLUGIN_NAME;
}

const char *TRTBEVPoolV2Creator::getPluginVersion() const noexcept {
    return TRTBEVPoolV2_PLUGIN_VERSION;
}

const PluginFieldCollection *TRTBEVPoolV2Creator::getFieldNames() noexcept {
    return &fc_;
}


} // namespace nvinfer1
