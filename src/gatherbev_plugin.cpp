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

void customGatherBEVPlugin(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
    const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream);

namespace nvinfer1
{
/********************注册PluginCreator*****************************/
REGISTER_TENSORRT_PLUGIN(GatherBEVPluginCreator);


/*********************静态变量的申明*******************************/
// class GatherBEVPluginCreator
PluginFieldCollection    GatherBEVPluginCreator::fc_ {};
std::vector<PluginField> GatherBEVPluginCreator::attr_;


// class GatherBEVPlugin
GatherBEVPlugin::GatherBEVPlugin(const std::string &name):
    name_(name){
}

GatherBEVPlugin::GatherBEVPlugin(const std::string &name, const void *buffer, size_t length):
    name_(name){
    memcpy(&m_, buffer, sizeof(m_));
}

GatherBEVPlugin::~GatherBEVPlugin(){
}

IPluginV2DynamicExt *GatherBEVPlugin::clone() const noexcept {
    auto p = new GatherBEVPlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t GatherBEVPlugin::getNbOutputs() const noexcept {
    return 1;
}
 
DataType GatherBEVPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, 
                                                                int32_t nbInputs) const noexcept {
    return inputTypes[0];  // 与adj_feat一致
}

DimsExprs GatherBEVPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, 
                                        int32_t nbInputs, IExprBuilder &exprBuilder) noexcept {
  // input[0] == adj_feat   b x 8 x 80 x 128 x 128
  // input[1] == curr_feat  b x 80 x 128 x 128
  // input[2] == flag       b x 1
  // out[0]                 b x (8+1)*80 x 128 x 128
    DimsExprs ret;
    ret.nbDims = inputs[0].nbDims - 1;

    IDimensionExpr const *n_feat = exprBuilder.operation(DimensionOperation::kSUM, 
                                                        *inputs[0].d[1],
                                                        *exprBuilder.constant(1));
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = exprBuilder.operation(DimensionOperation::kPROD, *n_feat, *inputs[0].d[2]);
    ret.d[2] = inputs[0].d[3];
    ret.d[3] = inputs[0].d[4];

    return ret; 
}

bool GatherBEVPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut,
                                                    int32_t nbInputs, int32_t nbOutputs) noexcept {
    // adj_feat
    if(pos == 0){
        return (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF) &&
                inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if(pos == 1){ // curr_feat
        return inOut[0].type == inOut[1].type && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if(pos == 3){ // out
        return inOut[0].type == inOut[3].type && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if(pos == 2){
        return inOut[pos].type == DataType::kINT32 && inOut[pos].format == TensorFormat::kLINEAR;
    }
    return false;
}

void GatherBEVPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, 
                                    const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept {
    return;
}

size_t GatherBEVPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, 
                                const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept {
    return 0;
}

int32_t GatherBEVPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc,
    const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept {

    customGatherBEVPlugin(inputDesc, outputDesc,
    inputs, outputs, workspace, stream);

    return 0;
}

void GatherBEVPlugin::destroy() noexcept {
    delete this;
    return;
}

int32_t GatherBEVPlugin::initialize() noexcept {
    return 0;
}

void GatherBEVPlugin::terminate() noexcept {
    return;
}

size_t GatherBEVPlugin::getSerializationSize() const noexcept {
    return sizeof(m_);
}

void GatherBEVPlugin::serialize(void *buffer) const noexcept {
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void GatherBEVPlugin::setPluginNamespace(const char *pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
    return;
}

const char *GatherBEVPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

const char *GatherBEVPlugin::getPluginType() const noexcept {
    return GATHERBEV_PLUGIN_NAME;
}

const char *GatherBEVPlugin::getPluginVersion() const noexcept {
    return GATHERBEV_PLUGIN_VERSION;
}

void GatherBEVPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, 
                                                        IGpuAllocator *gpuAllocator) noexcept {
    return;
}

void GatherBEVPlugin::detachFromContext() noexcept {
    return;
}

GatherBEVPluginCreator::GatherBEVPluginCreator() {
    attr_.clear();
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

GatherBEVPluginCreator::~GatherBEVPluginCreator() {
}


IPluginV2DynamicExt *GatherBEVPluginCreator::createPlugin(const char *name, 
                                    const PluginFieldCollection *fc) noexcept {
    GatherBEVPlugin *pObj = new GatherBEVPlugin(name);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

IPluginV2DynamicExt *GatherBEVPluginCreator::deserializePlugin(const char *name, 
                                        const void *serialData, size_t serialLength) noexcept {
    GatherBEVPlugin *pObj = new GatherBEVPlugin(name, serialData, serialLength);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

void GatherBEVPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
    return;
}

const char *GatherBEVPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

const char *GatherBEVPluginCreator::getPluginName() const noexcept {
    return GATHERBEV_PLUGIN_NAME;
}

const char *GatherBEVPluginCreator::getPluginVersion() const noexcept {
    return GATHERBEV_PLUGIN_VERSION;
}

const PluginFieldCollection *GatherBEVPluginCreator::getFieldNames() noexcept {
    return &fc_;
}


} // namespace nvinfer1
