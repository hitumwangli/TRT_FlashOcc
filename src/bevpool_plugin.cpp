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

// /* customBEVPoolPlugin的核函数接口部分 */
void customBEVPoolPlugin(
    const int channel, const int n_intervals, const int map_size, const int img_size,
    const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
    const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream);


namespace nvinfer1
{

/********************注册PluginCreator*****************************/
REGISTER_TENSORRT_PLUGIN(BEVPoolPluginCreator);


/*********************静态变量的申明*******************************/
// class BEVPoolPluginCreator
PluginFieldCollection    BEVPoolPluginCreator::fc_ {};
std::vector<PluginField> BEVPoolPluginCreator::attr_;


// class BEVPoolPlugin
BEVPoolPlugin::BEVPoolPlugin(const std::string &name, int bev_h, int bev_w, int n):
    name_(name){
    m_.bev_h = bev_h;
    m_.bev_w = bev_w;
    m_.n = n;
}

BEVPoolPlugin::BEVPoolPlugin(const std::string &name, const void *buffer, size_t length):
    name_(name){
    memcpy(&m_, buffer, sizeof(m_));
}

BEVPoolPlugin::~BEVPoolPlugin(){
}

IPluginV2DynamicExt *BEVPoolPlugin::clone() const noexcept {
    auto p = new BEVPoolPlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t BEVPoolPlugin::getNbOutputs() const noexcept {
    return 1;
}
 
DataType BEVPoolPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, 
                                                                int32_t nbInputs) const noexcept {
    return DataType::kFLOAT;
}

DimsExprs BEVPoolPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, 
                                        int32_t nbInputs, IExprBuilder &exprBuilder) noexcept {
    // input[0] depth            b*n x d x h x w
    // input[1] feat             b*n x c x h x w

    // input[2] ranks_depth      m
    // input[3] ranks_feat       m
    // input[4] ranks_bev        m
    // input[5] interval_starts  k
    // input[6] interval_lengths k

    // out[0]   bevfeat          b x c x h x w    

    DimsExprs ret;
    ret.nbDims = inputs[1].nbDims;
    ret.d[0] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[1].d[0], *exprBuilder.constant(m_.n)); 
    ret.d[1] = inputs[1].d[1];
    ret.d[2] = exprBuilder.constant(m_.bev_h);
    ret.d[3] = exprBuilder.constant(m_.bev_w);
    
    return ret; 
}

bool BEVPoolPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut,
                                                    int32_t nbInputs, int32_t nbOutputs) noexcept {
    // depth
    if(pos == 0){
        return (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF) &&
                 inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if(pos == 1){ // feat
        return inOut[0].type == inOut[1].type && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if(pos == 7){ // out
        return (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF) &&
                inOut[pos].format == TensorFormat::kLINEAR;
    }
    else{
        return inOut[pos].type == DataType::kINT32 && inOut[pos].format == TensorFormat::kLINEAR;
    }
    return false;
}

void BEVPoolPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, 
                                    const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept {
    return;
}

size_t BEVPoolPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, 
                                const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept {
    return 0;
}

int32_t BEVPoolPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc,
    const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
    // input[0] == depth            b*n x d x h x w
    // input[1] == feat             b*n x c x h x w
    // input[2] == ranks_depth      m
    // input[3] == ranks_feat       m
    // input[4] == ranks_bev        m
    // input[5] == interval_starts  k
    // input[6] == interval_lengths k

    int channel = inputDesc[1].dims.d[1];
    int n_intervals = inputDesc[5].dims.d[0];
    int map_size = m_.bev_h * m_.bev_w;
    int img_size = inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3];

    customBEVPoolPlugin(
    channel, n_intervals, map_size, img_size,
    inputDesc, outputDesc,
    inputs, outputs, workspace, stream);

    return 0;
}

void BEVPoolPlugin::destroy() noexcept {
    delete this;
    return;
}

int32_t BEVPoolPlugin::initialize() noexcept {
    return 0;
}

void BEVPoolPlugin::terminate() noexcept {
    return;
}

size_t BEVPoolPlugin::getSerializationSize() const noexcept {
    return sizeof(m_);
}

void BEVPoolPlugin::serialize(void *buffer) const noexcept {
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void BEVPoolPlugin::setPluginNamespace(const char *pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
    return;
}

const char *BEVPoolPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

const char *BEVPoolPlugin::getPluginType() const noexcept {
    return BEVPOOL_PLUGIN_NAME;
}

const char *BEVPoolPlugin::getPluginVersion() const noexcept {
    return BEVPOOL_PLUGIN_VERSION;
}

void BEVPoolPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, 
                                                        IGpuAllocator *gpuAllocator) noexcept {
    return;
}

void BEVPoolPlugin::detachFromContext() noexcept {
    return;
}

BEVPoolPluginCreator::BEVPoolPluginCreator() {
    attr_.clear();
    attr_.emplace_back(PluginField("bev_h", nullptr, PluginFieldType::kINT32, 1));
    attr_.emplace_back(PluginField("bev_w", nullptr, PluginFieldType::kINT32, 1));
    attr_.emplace_back(PluginField("n", nullptr, PluginFieldType::kINT32, 1));


    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

BEVPoolPluginCreator::~BEVPoolPluginCreator() {
}


IPluginV2DynamicExt *BEVPoolPluginCreator::createPlugin(const char *name, 
                                    const PluginFieldCollection *fc) noexcept {
    const PluginField *fields = fc->fields;

    int bev_h = -1;
    int bev_w = -1;
    int n = -1;

    for (int i = 0; i < fc->nbFields; ++i){
        if(std::string(fc->fields[i].name) == std::string("bev_h")){
            bev_h = *reinterpret_cast<const int *>(fc->fields[i].data);
        }
        else if(std::string(fc->fields[i].name) == std::string("bev_w")){
            bev_w = *reinterpret_cast<const int *>(fc->fields[i].data);
        }
        else if(std::string(fc->fields[i].name) == std::string("n")){
            n = *reinterpret_cast<const int *>(fc->fields[i].data);
        }
    }
    BEVPoolPlugin *pObj = new BEVPoolPlugin(name, bev_h, bev_w, n);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

IPluginV2DynamicExt *BEVPoolPluginCreator::deserializePlugin(const char *name, 
                                        const void *serialData, size_t serialLength) noexcept {
    BEVPoolPlugin *pObj = new BEVPoolPlugin(name, serialData, serialLength);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

void BEVPoolPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
    return;
}

const char *BEVPoolPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

const char *BEVPoolPluginCreator::getPluginName() const noexcept {
    return BEVPOOL_PLUGIN_NAME;
}

const char *BEVPoolPluginCreator::getPluginVersion() const noexcept {
    return BEVPOOL_PLUGIN_VERSION;
}

const PluginFieldCollection *BEVPoolPluginCreator::getFieldNames() noexcept {
    return &fc_;
}


} // namespace nvinfer1
