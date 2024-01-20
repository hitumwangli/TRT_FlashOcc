#ifndef __PREPROCESS_H__
#define __PREPROCESS_H__

#include "common.h"

void convert_RGBHWC_to_BGRCHW(uchar *input, uchar *output, 
                                                        int channels, int height, int width);

class PreprocessGPU{
public:
    PreprocessGPU(
        const int n_img, const int src_img_h, const int src_img_w, 
        const int dst_img_h, const int dst_img_w, const int type_int,
        const float *mean_cpu, const float *std_cpu,
        const float offset_h, const float offset_w, const float resize_radio
    );

    // void DoPreprocess(void ** const bev_buffer, std::vector<Box>& out_detections);
    void DoPreprocess(const uchar *inputs);
    ~PreprocessGPU();

public:
    float* mean;
    float* std;
    int n_img;
    int src_img_h;
    int src_img_w;
    int dst_img_h;
    int dst_img_w;
    int type_int;
    float offset_h;
    float offset_w;
    float resize_radio;
    float* imgs_dev_processed;
};

#endif