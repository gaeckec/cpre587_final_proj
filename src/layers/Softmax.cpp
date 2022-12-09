#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "Softmax.h"
#include <algorithm>
#include <math.h>
#include <chrono>

using namespace std::chrono;

extern float layer_times[12];
extern int layer_idx;

namespace ML {
    // --- Begin Student Code ---

    // Compute the convultion for the layer data
    void SoftMaxLayer::computeNaive(const LayerData &dataIn) const {
        // TODO: Your Code Here...
        Array1D_fp32 inputData = dataIn.getData<Array1D_fp32>();
        Array1D_fp32 outputData = getOutputData().getData<Array1D_fp32>();
        int X = getInputParams().dims[0];
        // printf("%d\n\r", I);

        auto start = high_resolution_clock::now();

        fp32 sum;
        for(int i = 0; i < X; ++i) {
            sum += exp(inputData[i]);
            // printf("%lf\n\r", sum);
        }

        for(int i = 0; i < X; ++i) {
            outputData[i] = exp(inputData[i]) / sum;
            // printf("%lf\n\r", outputData[i]);
        }

        auto end = high_resolution_clock::now();

        auto total = duration_cast<microseconds>(end - start);
        layer_times[layer_idx++] = total.count();
        // printf("SoftMax Finished in %d us\n\r", total.count());
    }


    // Compute the convolution using threads
    void SoftMaxLayer::computeThreaded(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the convolution using a tiled approach
    void SoftMaxLayer::computeTiled(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the convolution using SIMD
    void SoftMaxLayer::computeSIMD(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }
};