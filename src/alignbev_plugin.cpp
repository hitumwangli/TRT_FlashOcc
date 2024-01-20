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

#include "alignbev_plugin.h"
#include "common.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

void customAlignBEVPlugin(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
    const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream);


namespace nvinfer1 {
// class AlignBEVPlugin
AlignBEVPlugin::AlignBEVPlugin(const std::string &name):
    name_(name){
}

AlignBEVPlugin::AlignBEVPlugin(const std::string &name, const void *buffer, size_t length):
    name_(name){
    memcpy(&m_, buffer, sizeof(m_));
}

AlignBEVPlugin::~AlignBEVPlugin(){
}

IPluginV2DynamicExt *AlignBEVPlugin::clone() const noexcept {
    auto p = new AlignBEVPlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t AlignBEVPlugin::getNbOutputs() const noexcept {
    return 1;
}
 
DataType AlignBEVPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, 
                                                                int32_t nbInputs) const noexcept {
    return DataType::kFLOAT; // FIXME 
}

DimsExprs AlignBEVPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, 
                                        int32_t nbInputs, IExprBuilder &exprBuilder) noexcept {
    
    return inputs[0]; 
}

bool AlignBEVPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut,
                                                    int32_t nbInputs, int32_t nbOutputs) noexcept {
    // adj_feat  
    if(pos == 0){
        return (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF) &&
                inOut[pos].format == TensorFormat::kLINEAR;
    }    
    else if(pos == 2){ // out
        return (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF) &&
                inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if(pos == 1){ // transform
        return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }
    return false;
}

size_t AlignBEVPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, 
                                const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept {
    return 0;
}

int32_t AlignBEVPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc,
    const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
    
    customAlignBEVPlugin(inputDesc, outputDesc, inputs, outputs, workspace, stream);

    return 0;
}

void AlignBEVPlugin::destroy() noexcept {
    delete this;
    return;
}

void AlignBEVPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, 
                                    const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept {
    return;
}

int32_t AlignBEVPlugin::initialize() noexcept {
    return 0;
}

void AlignBEVPlugin::terminate() noexcept {
    return;
}

size_t AlignBEVPlugin::getSerializationSize() const noexcept {
    return sizeof(m_);
}

void AlignBEVPlugin::serialize(void *buffer) const noexcept {
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void AlignBEVPlugin::setPluginNamespace(const char *pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
    return;
}

const char *AlignBEVPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

const char *AlignBEVPlugin::getPluginType() const noexcept {
    return ALIGN_PLUGIN_NAME;
}

const char *AlignBEVPlugin::getPluginVersion() const noexcept {
    return ALIGN_PLUGIN_VERSION;
}

void AlignBEVPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, 
                                                        IGpuAllocator *gpuAllocator) noexcept {
    return;
}

void AlignBEVPlugin::detachFromContext() noexcept {
    return;
}

// class AlignBEVPluginCreator
PluginFieldCollection    AlignBEVPluginCreator::fc_ {};
std::vector<PluginField> AlignBEVPluginCreator::attr_;

AlignBEVPluginCreator::AlignBEVPluginCreator() {
    attr_.clear();
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

AlignBEVPluginCreator::~AlignBEVPluginCreator() {
}


IPluginV2DynamicExt *AlignBEVPluginCreator::createPlugin(const char *name, 
                                    const PluginFieldCollection *fc) noexcept {
    // const PluginField *fields = fc->fields;
    AlignBEVPlugin *pObj = new AlignBEVPlugin(name);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

IPluginV2DynamicExt *AlignBEVPluginCreator::deserializePlugin(const char *name, 
                                        const void *serialData, size_t serialLength) noexcept {
    AlignBEVPlugin *pObj = new AlignBEVPlugin(name, serialData, serialLength);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

void AlignBEVPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
    return;
}

const char *AlignBEVPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

const char *AlignBEVPluginCreator::getPluginName() const noexcept {
    return ALIGN_PLUGIN_NAME;
}

const char *AlignBEVPluginCreator::getPluginVersion() const noexcept {
    return ALIGN_PLUGIN_VERSION;
}

const PluginFieldCollection *AlignBEVPluginCreator::getFieldNames() noexcept {
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(AlignBEVPluginCreator);

} // namespace nvinfer1
