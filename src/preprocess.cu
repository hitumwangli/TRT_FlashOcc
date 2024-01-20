#include "common.h"
#include "preprocess.h"
#include <thrust/fill.h>
#include <fstream>

__global__ void convert_RGBHWC_to_BGRCHW_kernel(uchar *input, uchar *output, 
                                                            int channels, int height, int width){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < channels * height * width){
        int y = index / 3 / width;
        int x = index / 3 % width;
        int c = 2 - index % 3;  // RGB to BGR

        output[c * height * width + y * width + x] = input[index];
    }
}
// RGBHWC to BGRCHW
void convert_RGBHWC_to_BGRCHW(uchar *input, uchar *output, 
                                                        int channels, int height, int width){
    convert_RGBHWC_to_BGRCHW_kernel<<<DIVUP(channels * height * width, NUM_THREADS), NUM_THREADS>>>
                                                            (input, output, channels, height, width);
}



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
    
    // printf("mean[0]: %f \n", mean[0]);
    // printf("mean[1]: %f \n", mean[1]);
    // printf("mean[2]: %f \n", mean[2]);
    // printf("std[0]: %f \n", std[0]);
    // printf("std[1]: %f \n", std[1]);
    // printf("std[2]: %f \n", std[2]);

    int i = (idx / n) / dst_w;
    int j = (idx / n) % dst_w;
    int k = idx % n;

	int pX = (int) roundf((i / radio_h) + offset_h);
	int pY = (int) roundf((j / radio_w) + offset_w);
 
	if (pX < src_h && pX >= 0 && pY < src_w && pY >= 0){
        int s1 = k * src_img_step + 0 * src_img_step / 3 + pX * src_row_step + pY; // save as 6,3,256,704
        int s2 = k * src_img_step + 1 * src_img_step / 3 + pX * src_row_step + pY;
        int s3 = k * src_img_step + 2 * src_img_step / 3 + pX * src_row_step + pY;

        // printf("mode_1");
        int d1 = k * dst_img_step + 0 * dst_img_step / 3 + i * dst_row_step + j; // save as 6,3,256,704
        int d2 = k * dst_img_step + 1 * dst_img_step / 3 + i * dst_row_step + j;
        int d3 = k * dst_img_step + 2 * dst_img_step / 3 + i * dst_row_step + j;
        // printf("mode_2");
        // int d3 = k * dst_img_step + 0 * dst_img_step / 3 + i * dst_row_step + j;
        // int d2 = k * dst_img_step + 1 * dst_img_step / 3 + i * dst_row_step + j;
        // int d1 = k * dst_img_step + 2 * dst_img_step / 3 + i * dst_row_step + j;
        // printf("mode_3");
        // int d1 = k * dst_img_step + 0 + 3*(i * dst_row_step + j); // save as 6,256,704,3. RGB
        // int d2 = k * dst_img_step + 1 + 3*(i * dst_row_step + j);
        // int d3 = k * dst_img_step + 2 + 3*(i * dst_row_step + j);
        // printf("mode_4");
        // int d3 = k * dst_img_step + 0 + 3*(i * dst_row_step + j); // save as 6,256,704,3. BGR
        // int d2 = k * dst_img_step + 1 + 3*(i * dst_row_step + j);
        // int d1 = k * dst_img_step + 2 + 3*(i * dst_row_step + j);

		dst_dev[d1] = (static_cast<T>(src_dev[s1]) - static_cast<T>(mean[0])) / static_cast<T>(std[0]);
		dst_dev[d2] = (static_cast<T>(src_dev[s2]) - static_cast<T>(mean[1])) / static_cast<T>(std[1]);
		dst_dev[d3] = (static_cast<T>(src_dev[s3]) - static_cast<T>(mean[2])) / static_cast<T>(std[2]);

        // // printf("idx(%d), n(%d), dst_w(%d), dst_h(%d), \
        // //        s1(%d) = k(%d)*src_img_step(%d) + 0*src_img_step(%d)/3 + pX(%d)*src_row_step(%d) + pY(%d), \
        // //        || d1(%d) = k(%d)*dst_img_step(%d) + 0*dst_img_step(%d)/3 + i(%d)*dst_row_step(%d) + j(%d), \
        // //        || idx:%d, s1:%d,  src_dev[s1]:%f, d1:%d,  dst_dev[d1]:%f \
        // //        \n", \
        // //     idx, n, dst_w, dst_h, \
        // //     s1, k, src_img_step, src_img_step, pX, src_row_step, pY, \
        // //     d1, k, dst_img_step, dst_img_step, i, dst_row_step, j, \
        // //     idx, s1, (float)src_dev[s1], d1, (float)dst_dev[d1] \
        // // );

        // printf("\
        //        || idx:%d, s1:%d,  src_dev[s1]:%f, d1:%d,  dst_dev[d1]:%f \
        //        \n", \
        //     idx, s1, (float)src_dev[s1], d1, (float)dst_dev[d1] \
        // );
	}


}

PreprocessGPU::PreprocessGPU(
    const int n_img, const int src_img_h, const int src_img_w, 
    const int dst_img_h, const int dst_img_w, const int type_int,
    const float *mean_cpu, const float *std_cpu,
    const float offset_h, const float offset_w, const float resize_radio
):
    n_img(n_img),
    src_img_h(src_img_h),
    src_img_w(src_img_w),
    dst_img_h(dst_img_h),
    dst_img_w(dst_img_w),
    type_int(type_int),
    offset_h(offset_h),
    offset_w(offset_w),
    resize_radio(resize_radio)
{
    cudaMalloc((void**)&mean, 3 * sizeof(float));
    cudaMemcpy(mean, mean_cpu, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&std, 3 * sizeof(float));
    cudaMemcpy(std, std_cpu, 3 * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA(cudaMalloc((void**)&(imgs_dev_processed), n_img * dst_img_h * dst_img_w * 3 * sizeof(float)));
}

PreprocessGPU::~PreprocessGPU(){
    CHECK_CUDA(cudaFree(imgs_dev_processed));
}

void PreprocessGPU::DoPreprocess(const uchar *inputs){

    int src_row_step = src_img_w;
    int dst_row_step = dst_img_w;

    int src_img_step = src_img_w * src_img_h * 3;
    int dst_img_step = dst_img_w * dst_img_h * 3;

    dim3 grid(DIVUP(dst_img_h * dst_img_w * n_img,  NUM_THREADS));
    dim3 block(NUM_THREADS);

    int DataType_kFLOAT_int = int(nvinfer1::DataType::kFLOAT);  // 0
    int DataType_kHALF_int = int(nvinfer1::DataType::kHALF);    // 1

    float *outputs = imgs_dev_processed;

    if (type_int == DataType_kFLOAT_int)
    {
        // preprocess_kernel<<<grid, block, 0, stream>>>(
        preprocess_kernel<<<grid, block>>>(
                                                reinterpret_cast<const uint8_t *>(inputs),
                                                reinterpret_cast<float *>(outputs),
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
                                                reinterpret_cast<const float *>(mean), 
                                                reinterpret_cast<const float *>(std),
                                                dst_img_h, 
                                                dst_img_w,
                                                n_img);
    }
    else if (type_int == DataType_kHALF_int)
    {
        // printf("pre : half\n");
        // preprocess_kernel<<<grid, block, 0, stream>>>(
        preprocess_kernel<<<grid, block>>>(
                                                reinterpret_cast<const uint8_t *>(inputs),
                                                reinterpret_cast<__half *>(outputs),
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
                                                reinterpret_cast<const float *>(mean), 
                                                reinterpret_cast<const float *>(std),
                                                dst_img_h, 
                                                dst_img_w,
                                                n_img);

    }
    else
    {
        printf("\tUnsupport datatype!\n");
    }
}