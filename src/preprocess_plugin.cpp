#include "preprocess_plugin.h"
#include <map>
#include <cstring>
#include "common.h"

// /* customPreprocessPlugin的核函数接口部分 */
void customPreprocessPlugin(const int n_img, const int src_img_h, const int src_img_w, 
    const int dst_img_h, const int dst_img_w, const int type_int,
    const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream,
    const float offset_h, const float offset_w, const float resize_radio);

namespace nvinfer1
{

/********************注册PluginCreator*****************************/
REGISTER_TENSORRT_PLUGIN(PreprocessPluginCreator);


/*********************静态变量的申明*******************************/
// class PreprocessPluginCreator
PluginFieldCollection    PreprocessPluginCreator::mFC {};
std::vector<PluginField> PreprocessPluginCreator::mAttrs;


/*********************CustomScalarPlugin实现部分***********************/
// class PreprocessPlugin
PreprocessPlugin::PreprocessPlugin(const std::string &name, int crop_h, int crop_w, float resize_radio):
    mName(name){
    mParams.crop_h = crop_h;
    mParams.crop_w = crop_w;
    mParams.resize_radio = resize_radio;
}

PreprocessPlugin::PreprocessPlugin(const std::string &name, const void *buffer, size_t length):
    mName(name){
    memcpy(&mParams, buffer, sizeof(mParams));
}

PreprocessPlugin::~PreprocessPlugin(){
    /* 这里的析构函数不需要做任何事情，生命周期结束的时候会自动调用terminate和destroy */
}

IPluginV2DynamicExt *PreprocessPlugin::clone() const noexcept {
    auto p = new PreprocessPlugin(mName, &mParams, sizeof(mParams));
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

int32_t PreprocessPlugin::getNbOutputs() const noexcept {
    /* 一般来说所有插件的实现差不多一致 */
    return 1;
}
 
DataType PreprocessPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, 
                                                                int32_t nbInputs) const noexcept {
    // return DataType::kHALF;
    return DataType::kFLOAT;
}

DimsExprs PreprocessPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, 
                                        int32_t nbInputs, IExprBuilder &exprBuilder) noexcept {
    int input_h = inputs[0].d[2]->getConstantValue();
    int input_w = inputs[0].d[3]->getConstantValue(); // * 4;

    int output_h = input_h * mParams.resize_radio - mParams.crop_h;
    int output_w = input_w * mParams.resize_radio - mParams.crop_w;

    DimsExprs ret;
    ret.nbDims = inputs[0].nbDims;
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = inputs[0].d[1];
    ret.d[2] =  exprBuilder.constant(output_h);
    ret.d[3] =  exprBuilder.constant(output_w);
    
    return ret; 
}

bool PreprocessPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut,
                                                    int32_t nbInputs, int32_t nbOutputs) noexcept {
    bool res;
    switch (pos)
    {
    case 0: // images
        res = (inOut[0].type == DataType::kINT8 || inOut[0].type == DataType::kINT32) && 
                inOut[0].format == TensorFormat::kLINEAR;
        break;
    case 1: // mean
        res = (inOut[1].type == DataType::kFLOAT) &&
                inOut[1].format == TensorFormat::kLINEAR;
        break;
    case 2: // std
        res = (inOut[2].type == DataType::kFLOAT) &&
                inOut[2].format == TensorFormat::kLINEAR;
        break;
    case 3: // 输出 img tensor
        res = (inOut[3].type == DataType::kFLOAT || inOut[3].type == DataType::kHALF) && 
                inOut[3].format == inOut[0].format;

        // res = inOut[3].type == DataType::kHALF && inOut[3].format == inOut[0].format;
        break;
    default: 
        res = false;
    }
    return res;
}

void PreprocessPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, 
                                    const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept {
    return;
}

size_t PreprocessPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, 
                                const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept {
    return 0;
}

int32_t PreprocessPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc,
    const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
    /*
     * Plugin的核心的地方。每个插件都有一个自己的定制方案
     * Plugin直接调用kernel的地方
    */
    float offset_h = mParams.crop_h / mParams.resize_radio;
    float offset_w = mParams.crop_w / mParams.resize_radio;
    float resize_radio = mParams.resize_radio;

    int n_img = inputDesc[0].dims.d[0];
    int src_img_h = inputDesc[0].dims.d[2];
    int src_img_w = inputDesc[0].dims.d[3]; // * 4;
    
    int dst_img_h = outputDesc[0].dims.d[2];
    int dst_img_w = outputDesc[0].dims.d[3];
    int type_int = int(outputDesc[0].type);


    customPreprocessPlugin(n_img, src_img_h, src_img_w, 
    dst_img_h, dst_img_w, type_int,
    inputs, outputs, workspace, stream,
    offset_h, offset_w, resize_radio);
    return 0;
}

void PreprocessPlugin::destroy() noexcept {
    delete this;
    return;
}

int32_t PreprocessPlugin::initialize() noexcept {
    return 0;
}

void PreprocessPlugin::terminate() noexcept {
    return;
}

size_t PreprocessPlugin::getSerializationSize() const noexcept {
    /* 如果把所有的参数给放在mParams中的话, 一般来说所有插件的实现差不多一致 */
    return sizeof(mParams);
}

void PreprocessPlugin::serialize(void *buffer) const noexcept {
    memcpy(buffer, &mParams, sizeof(mParams));
    return;
}

void PreprocessPlugin::setPluginNamespace(const char *pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
    return;
}

const char *PreprocessPlugin::getPluginNamespace() const noexcept {
    /* 所有插件实现都差不多 */
    return mNamespace.c_str();
}

const char *PreprocessPlugin::getPluginType() const noexcept {
    /* 一般来说所有插件的实现差不多一致 */
    return bevdet::PRE_PLUGIN_NAME;
}

const char *PreprocessPlugin::getPluginVersion() const noexcept {
    return bevdet::PRE_PLUGIN_VERSION;
}

void PreprocessPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, 
                                                        IGpuAllocator *gpuAllocator) noexcept {
    return;
}

void PreprocessPlugin::detachFromContext() noexcept {
    return;
}

PreprocessPluginCreator::PreprocessPluginCreator() {
    
    mAttrs.clear();
    mAttrs.emplace_back(PluginField("crop_h", nullptr, PluginFieldType::kINT32, 1));
    mAttrs.emplace_back(PluginField("crop_w", nullptr, PluginFieldType::kINT32, 1));
    mAttrs.emplace_back(PluginField("resize_radio", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mAttrs.size();
    mFC.fields   = mAttrs.data();
}

PreprocessPluginCreator::~PreprocessPluginCreator() {
}


IPluginV2DynamicExt *PreprocessPluginCreator::createPlugin(const char *name, 
                                    const PluginFieldCollection *fc) noexcept {
    /*
     * 通过Creator创建一个Plugin的实现，这个时候会通过mFC中取出需要的参数, 并实例化一个Plugin
     * 这个案例中，参数有scalar和scale两个参数。从fc中取出来对应的数据来初始化这个plugin
    */
    const PluginField *fields = fc->fields;

    int crop_h = -1;
    int crop_w = -1;
    float resize_radio = 0.f;

    for (int i = 0; i < fc->nbFields; ++i){
        if(std::string(fc->fields[i].name) == std::string("crop_h")){
            crop_h = *reinterpret_cast<const int *>(fc->fields[i].data);
        }
        else if(std::string(fc->fields[i].name) == std::string("crop_w")){
            crop_w = *reinterpret_cast<const int *>(fc->fields[i].data);
        }
        else if(std::string(fc->fields[i].name) == std::string("resize_radio")){
            resize_radio = *reinterpret_cast<const float *>(fc->fields[i].data);
        }
    }
    PreprocessPlugin *pObj = new PreprocessPlugin(name, crop_h, crop_w, resize_radio);
    pObj->setPluginNamespace(mNamespace.c_str());
    return pObj;
}

IPluginV2DynamicExt *PreprocessPluginCreator::deserializePlugin(const char *name, 
                                        const void *serialData, size_t serialLength) noexcept {
    /* 反序列化插件其实就是实例化一个插件，所有插件实现都差不多 */
    PreprocessPlugin *pObj = new PreprocessPlugin(name, serialData, serialLength);
    pObj->setPluginNamespace(mNamespace.c_str());
    return pObj;
}

void PreprocessPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept {
    /* 所有插件实现都差不多 */
    mNamespace = pluginNamespace;
    return;
}

const char *PreprocessPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

const char *PreprocessPluginCreator::getPluginName() const noexcept {
    return bevdet::PRE_PLUGIN_NAME;
}

const char *PreprocessPluginCreator::getPluginVersion() const noexcept {
    /* 一般来说所有插件的实现差不多一致 */
    return bevdet::PRE_PLUGIN_VERSION;
}

const PluginFieldCollection *PreprocessPluginCreator::getFieldNames() noexcept {
    /* 所有插件实现都差不多 */
    return &mFC;
}

} // namespace nvinfer1
