#include <iostream>
#include <fstream>
#include "debug.h"

void SavePtrTxt(std::string filepath, uint8_t *out_data, int count)
{
    FILE *fp_blob = fopen(filepath.c_str(), "w");
    for (int i = 0; i < count; i++)
    {
        fprintf(fp_blob, "%d\n", out_data[i]);
    }
    fclose(fp_blob);
}

void SavePtrTxt(std::string filepath, float *out_data, int count)
{
    FILE *fp_blob = fopen(filepath.c_str(), "w");
    for (int i = 0; i < count; i++)
    {
        fprintf(fp_blob, "%.18e\n", out_data[i]);
    }
    fclose(fp_blob);
}

void SavePtrTxt(std::string filepath, int *out_data, int count)
{
    FILE *fp_blob = fopen(filepath.c_str(), "w");
    for (int i = 0; i < count; i++)
    {
        fprintf(fp_blob, "%d\n", out_data[i]);
    }
    fclose(fp_blob);
}


void ReadPtrTxt(std::string filepath, float *out_data, int count)
{
    FILE *fp_blob = fopen(filepath.c_str(), "r");
    for (int i = 0; i < count; i++)
    {
        fscanf(fp_blob, "%f\n", &out_data[i]);
    }
    fclose(fp_blob);
}

void ReadPtrTxt(std::string filepath, int *out_data, int count)
{
    FILE *fp_blob = fopen(filepath.c_str(), "r");
    for (int i = 0; i < count; i++)
    {
        fscanf(fp_blob, "%d\n", &out_data[i]);
    }
    fclose(fp_blob);
}



