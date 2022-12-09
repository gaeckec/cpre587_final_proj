#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "Flatten.h"
#include <algorithm>
#include <chrono>

using namespace std::chrono;

extern float layer_times[12];
extern int layer_idx;

namespace ML {
    // --- Begin Student Code ---

    // Compute the convultion for the layer data
    void FlattenLayer::computeNaive(const LayerData &dataIn) const {
        // TODO: Your Code Here...
        Array3D_fp32 inputData = dataIn.getData<Array3D_fp32>();

        int DEPTH = getInputParams().dims[0];
        int HEIGHT = getInputParams().dims[1];
        int WIDTH  = getInputParams().dims[2];

        Array1D_fp32 outputData = getOutputData().getData<Array1D_fp32>();

        auto start = high_resolution_clock::now();

        for(int i = 0; i < DEPTH; ++i) {
            for(int j = 0; j < HEIGHT; ++j) {
                for(int k = 0; k < WIDTH; ++k) {
                    int idx = (WIDTH*HEIGHT*i) + (WIDTH*j) + k;
                    // printf("%d\n\r", idx);
                    outputData[idx] = inputData[i][j][k];
                }
            }
        }
        auto end = high_resolution_clock::now();

        auto total = duration_cast<microseconds>(end - start);
        layer_times[layer_idx++] = total.count();
        // printf("Flatten Finished in %d us\n\r", total.count());
    }


    // Compute the convolution using threads
    void FlattenLayer::computeThreaded(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the convolution using a tiled approach
    void FlattenLayer::computeTiled(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the convolution using SIMD
    void FlattenLayer::computeSIMD(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }
};