#include <iostream>
#include <fstream>

void SavePtrTxt(std::string filepath, uint8_t *out_data, int count);

void SavePtrTxt(std::string filepath, float *out_data, int count);

void SavePtrTxt(std::string filepath, int32_t *out_data, int count);

void ReadPtrTxt(std::string filepath, float *out_data, int count);

void ReadPtrTxt(std::string filepath, int32_t *out_data, int count);

