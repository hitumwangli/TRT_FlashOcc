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

#include <NvInfer.h>
#include <cmath>
#include <string>
#include <vector>
#include <unistd.h>
#include <iostream>


namespace bevdet                                     // 对应于torch中注册onnx 算子时限定的域名
{
static const char *PRE_PLUGIN_NAME {"Preprocess"};   // 对应于torch中注册onnx 算子时限定的算子名称
static const char *PRE_PLUGIN_VERSION {"1"};         // 版本号 无限定
} // namespace

namespace nvinfer1
{
    /* 
* 在这里面需要创建两个类, 一个是普通的Plugin类, 一个是PluginCreator类
*  - Plugin类是插件类，用来写插件的具体实现
*  - PluginCreator类是插件工厂类，用来根据需求创建插件。调用插件是从这里走的
*/
class PreprocessPlugin : public IPluginV2DynamicExt
{
private:
    const std::string mName;
    std::string       mNamespace;
    struct
    {
        int crop_h;
        int crop_w;
        float resize_radio;
    } mParams; // 当这个插件op需要有参数的时候，把这些参数定义为成员变量，可以单独拿出来定义，也可以像这样定义成一个结构体

public:
    /*
     * 我们在编译的过程中会有大概有三次创建插件实例的过程
     * 1. parse阶段: 第一次读取onnx来parse这个插件。会读取参数信息并转换为TensorRT格式
     * 2. clone阶段: parse完了以后，TensorRT为了去优化这个插件会复制很多副本出来来进行很多优化测试。也可以在推理的时候供不同的context创建插件的时候使用
     * 3. deseriaze阶段: 将序列化好的Plugin进行反序列化的时候也需要创建插件的实例
    */
    PreprocessPlugin() = delete;                                                             //默认构造函数，一般直接delete
    PreprocessPlugin(const std::string &name, int crop_h, int crop_w, float resize_radio);   //parse, clone时候用的构造函数
    PreprocessPlugin(const std::string &name, const void *buffer, size_t length);            //反序列化的时候用的构造函数
    ~PreprocessPlugin();

    // Method inherited from IPluginV2
    const char *getPluginType() const noexcept override;
    const char *getPluginVersion() const noexcept override;
    int32_t     getNbOutputs() const noexcept override;
    int32_t     initialize() noexcept override;
    void        terminate() noexcept override;
    size_t      getSerializationSize() const noexcept override;
    void        serialize(void *buffer) const noexcept override;
    void        destroy() noexcept override;
    void        setPluginNamespace(const char *pluginNamespace) noexcept override;
    const char *getPluginNamespace() const noexcept override;

    // Method inherited from IPluginV2Ext
    DataType getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept override;
    void     attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept override;
    void     detachFromContext() noexcept override;

    // Method inherited from IPluginV2DynamicExt
    IPluginV2DynamicExt *clone() const noexcept override;
    DimsExprs            getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override;
    bool                 supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void                 configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override;
    size_t               getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override;
    int32_t              enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

protected:
    // To prevent compiler warnings
    using nvinfer1::IPluginV2::enqueue;
    using nvinfer1::IPluginV2::getOutputDimensions;
    using nvinfer1::IPluginV2::getWorkspaceSize;
    using nvinfer1::IPluginV2Ext::configurePlugin;
};

class PreprocessPluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection    mFC;           //接受plugionFields传进来的权重和参数，并将信息传递给Plugin，内部通过createPlugin来创建带参数的plugin
    static std::vector<PluginField> mAttrs;        //用来保存这个插件op所需要的权重和参数, 从onnx中获取, 同样在parse的时候使用
    std::string                     mNamespace;

public:
    PreprocessPluginCreator();  //初始化mFC以及mAttrs
    ~PreprocessPluginCreator();
    const char*                  getPluginName() const noexcept override;
    const char*                  getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
    const char*                  getPluginNamespace() const noexcept override;
    IPluginV2DynamicExt *        createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;  //通过包含参数的mFC来创建Plugin。调用上面的Plugin的构造函数
    IPluginV2DynamicExt *        deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void                         setPluginNamespace(const char* pluginNamespace) noexcept override;
};

} // namespace nvinfer1
