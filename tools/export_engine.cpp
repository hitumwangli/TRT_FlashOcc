#include <NvInfer.h>
#include <NvOnnxParser.h>
// #include <dlfcn.h>
#include <fstream>
#include <chrono>
#include <unistd.h>
#include <cstring>
#include "common.h"
#include "preprocess_plugin.h"
#include "alignbev_plugin.h"
#include "bevpool_plugin.h"
#include "gatherbev_plugin.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;
using namespace nvinfer1;

static Logger     gLogger(ILogger::Severity::kERROR);


int main(int argc, char * argv[]){

    if(argc != 3){
        printf("./export model.onnx model.engine\n");
        return 0;
    }
    std::string onnxFile(argv[1]);
    std::string trtFile(argv[2]);

    CHECK_CUDA(cudaSetDevice(0));
    ICudaEngine *engine = nullptr;
    std::vector<Dims32> min_shapes{
        {4, {6, 3, 256, 704}},
        // {1, {3}},
        // {1, {3}},
        // {3, {1, 6, 27}},
        {1, {200000}},
        {1, {200000}},
        {1, {200000}},
        {1, {8000}},
        {1, {8000}},
        // {5, {1, 8, 80, 128, 128}},
        // {3, {1, 8, 6}},
        // {2, {1, 1}}
    };

    std::vector<Dims32> opt_shapes{
        {4, {6, 3, 256, 704}},
        // {1, {3}},
        // {1, {3}},
        // {3, {1, 6, 27}},
        {1, {356760}},
        {1, {356760}},
        {1, {356760}},
        {1, {13360}},
        {1, {13360}},
        // {5, {1, 8, 80, 128, 128}},
        // {3, {1, 8, 6}},
        // {2, {1, 1}}
    };

    std::vector<Dims32> max_shapes{
        {4, {6, 3, 256, 704}},
        // {1, {3}},
        // {1, {3}},
        // {3, {1, 6, 27}},
        {1, {370000}},
        {1, {370000}},
        {1, {370000}},
        {1, {30000}},
        {1, {30000}},
        // {5, {1, 8, 80, 128, 128}},
        // {3, {1, 8, 6}},
        // {2, {1, 1}}
    };

    if (access(trtFile.c_str(), F_OK) == 0)
    {
        std::ifstream engineFile(trtFile, std::ios::binary);
        long int      fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0)
        {
            std::cout << "Failed getting serialized engine!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr)
        {
            std::cout << "Failed loading engine!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded loading engine!" << std::endl;
    }
    else
    {
        // 定义基本组件
        IBuilder *            builder = createInferBuilder(gLogger);
        INetworkDefinition *  network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IBuilderConfig *      config  = builder->createBuilderConfig();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 3UL << 32UL);
        // if (bFP16Mode)
        // {
            config->setFlag(BuilderFlag::kFP16);
            // config->setFlag(BuilderFlag::kINT8);
        // }

        // 通过onnx解析器解析的结果会以类似addConv的方式填充到network中
        nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, gLogger);
        if (!parser->parseFromFile(onnxFile.c_str(), int(gLogger.reportable_severity)))
        {
            std::cout << std::string("Failed parsing .onnx file!") << std::endl;
            for (int i = 0; i < parser->getNbErrors(); ++i)
            {
                auto *error = parser->getError(i);
                std::cout << std::to_string(int(error->code())) << std::string(":") << std::string(error->desc()) << std::endl;
            }
            return 1;
        }
        std::cout << std::string("Succeeded parsing .onnx file!") << std::endl;

        // 如果模型有多个输入，则必须多个profile
        IOptimizationProfile *profile = builder->createOptimizationProfile();
        // 配置输入的最小、最优、最大的范围
        for(size_t i = 0; i < min_shapes.size(); i++){
            ITensor *it = network->getInput(i);
            profile->setDimensions(it->getName(), OptProfileSelector::kMIN, min_shapes[i]);
            profile->setDimensions(it->getName(), OptProfileSelector::kOPT, opt_shapes[i]);
            profile->setDimensions(it->getName(), OptProfileSelector::kMAX, max_shapes[i]);
        }
        config->addOptimizationProfile(profile);

        // 根据指定好的配置构建引擎，得到序列化模型engineString
        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        if (engineString == nullptr || engineString->size() == 0)
        {
            std::cout << "Failed building serialized engine!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded building serialized engine!" << std::endl;

        // 反序列化后得到engine
        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr)
        {
            std::cout << "Failed building engine!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded building engine!" << std::endl;

        // 保存engine
        std::ofstream engineFile(trtFile, std::ios::binary);
        if (!engineFile)
        {
            std::cout << "Failed opening file to write" << std::endl;
            return 1;
        }
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        if (engineFile.fail())
        {
            std::cout << "Failed saving .plan file!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded saving .plan file!" << std::endl;
    }
    // 执行推理
    // 输出可执行context
    IExecutionContext *context = engine->createExecutionContext();

    for(size_t i = 0; i < min_shapes.size(); i++){
        context->setBindingDimensions(i, min_shapes[i]);  // 利用setBindingDimensions来设置相关输入尺寸
    }

    std::cout << std::string("Binding all? ") << std::string(context->allInputDimensionsSpecified() ? "Yes" : "No") << std::endl;
    int nBinding = engine->getNbBindings();               // 获取网络输入输出数量
    int nInput   = 0;
    for (int i = 0; i < nBinding; ++i)
    {
        nInput += int(engine->bindingIsInput(i));         // 累计网络输入数量
    }
    //int nOutput = nBinding - nInput;                    // 网络输出数量
    for (int i = 0; i < nBinding; ++i)
    {
        std::cout << std::string("Bind[") << i << std::string(i < nInput ? "]:i[" : "]:o[") << (i < nInput ? i : i - nInput) << std::string("]->");
        std::cout << dataTypeToString(engine->getBindingDataType(i)) << std::string(" ");
        std::cout << shapeToString(context->getBindingDimensions(i)) << std::string(" ");
        std::cout << engine->getBindingName(i) << std::endl;
    }

    // 打印输入输出的大小
    std::vector<int> vBindingSize(nBinding, 0);
    for (int i = 0; i < nBinding; ++i)
    {
        Dims32 dim  = context->getBindingDimensions(i);
        int size = 1;
        for (int j = 0; j < dim.nbDims; ++j)
        {
            size *= dim.d[j];
        }
        vBindingSize[i] = size * dataTypeToSize(engine->getBindingDataType(i));
        printf("id : %d, %d\n", i, vBindingSize[i]);
    }

    std::vector<void *> vBufferH {nBinding, nullptr};
    std::vector<void *> vBufferD {nBinding, nullptr};
    for (int i = 0; i < nBinding; ++i)
    {
        vBufferH[i] = (void *)new char[vBindingSize[i]]; 
        memset(vBufferH[i], 0, vBindingSize[i]); // FIXME
        CHECK_CUDA(cudaMalloc(&vBufferD[i], vBindingSize[i]));
    }

    for (int i = 0; i < nInput; ++i)
    {
        CHECK_CUDA(cudaMemcpy(vBufferD[i], vBufferH[i], vBindingSize[i], cudaMemcpyHostToDevice));
    }

    context->executeV2(vBufferD.data()); // 同步，如果需要异步执行，请用context->enqueueV2(buffers, stream, nullptr);

    // auto start = high_resolution_clock::now();
    // for(int i = 0; i < 100; i++){
    //     context->executeV2(vBufferD.data());
    // }
    // auto end = high_resolution_clock::now();
    // duration<double> t = end - start;
    // printf("infer : %.4lf\n", t.count() * 1000 / 100);

    // 将输出tensor从device拷贝到host
    for (int i = nInput; i < nBinding; ++i)
    {
        CHECK_CUDA(cudaMemcpy(vBufferH[i], vBufferD[i], vBindingSize[i], cudaMemcpyDeviceToHost));
    }

    for (int i = 0; i < nBinding; ++i)
    {
        CHECK_CUDA(cudaFree(vBufferD[i]));
    }

    return 0;
}