#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "MaxPooling.h"
#include <algorithm>
#include <chrono>

using namespace std::chrono;

extern float layer_times[12];
extern int layer_idx;

namespace ML {
    // --- Begin Student Code ---

    // Compute the convultion for the layer data
    void MaxPoolingLayer::computeNaive(const LayerData &dataIn) const {
        // TODO: Your Code Here...
        Array3D_fp32 inputData = dataIn.getData<Array3D_fp32>();

        int X = getInputParams().dims[0];
        int Y = getInputParams().dims[1];
        int Z  = getInputParams().dims[2];

        Array3D_fp32 outputData = getOutputData().getData<Array3D_fp32>();

        auto start = high_resolution_clock::now();

        for(int i = 0; i < X; i += 2) {
            for(int j = 0; j < Y; j += 2) {
                for(int k = 0; k < Z; ++k) {
                    outputData[i/2][j/2][k] = std::max(std::max(inputData[i][j][k], inputData[i+1][j][k]), std::max(inputData[i][j+1][k], inputData[i+1][j+1][k]));
                    // printf("%lf %lf\n\r%lf %lf\n\r%lf\n\r", inputData[i][j][k], inputData[i+1][j][k], inputData[i][j+1][k], inputData[i+1][j+1][k], outputData[i/2][j/2][k]);
                }
            }
        }
        auto end = high_resolution_clock::now();

        auto total = duration_cast<microseconds>(end - start);
        layer_times[layer_idx++] = total.count();
        // printf("MaxPooling Finished in %d us\n\r", total.count());
    }


    // Compute the convolution using threads
    void MaxPoolingLayer::computeLinearQ(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        computeNaive(dataIn);
    }


    // Compute the convolution using a tiled approach
    void MaxPoolingLayer::computeLogQ(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        computeNaive(dataIn);
    }


    // Compute the convolution using SIMD
    void MaxPoolingLayer::computeHash(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        computeNaive(dataIn);
    }
};