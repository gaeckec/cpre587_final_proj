#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "Dense.h"
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

extern float layer_times[12];
extern int layer_idx;

namespace ML {
    // --- Begin Student Code ---

    // Compute the full connected for the layer data
    void DenseLayer::computeNaive(const LayerData &dataIn) const {
        //Define Parameters
        int input_height = getInputParams().dims[0];
        int num_input_channels = getWeightParams().dims[1];

        //Probably have a variable assicated with this        
        int batch_size = 1;

        //Preset LayerDate Tpye
        LayerData Weight_data = getWeightData();
        LayerData Bias_data = getBiasData();
        LayerData Output_data = getOutputData();

        //Map values to memory
        Array2D_fp32 denseWeightData = Weight_data.getData<Array2D_fp32>();
        Array1D_fp32 denseBiasData = Bias_data.getData<Array1D_fp32>();
        Array1D_fp32 denseInputData = dataIn.getData<Array1D_fp32>();
        Array1D_fp32 denseOutputData = Output_data.getData<Array1D_fp32>();

        //predeclair variables
        int n,m,h;

        auto start = high_resolution_clock::now();
        //loop through and perform the opperation
        for(n = 0; n < batch_size; n++){
            for(m = 0; m < num_input_channels; m++){
                for(h = 0; h < input_height; h++) {                                    
                    denseOutputData[m] += denseInputData[h] * denseWeightData[h][m];                           
                }
                
                denseOutputData[m] += denseBiasData[m];    
                if(denseOutputData[m] < 0) { denseOutputData[m] = 0.0; } 
            } 
        }

        auto end = high_resolution_clock::now();
        auto total = duration_cast<microseconds>(end - start);
        layer_times[layer_idx++] = total.count();
    }

    // Compute the Dense using threads
    void DenseLayer::computeLinearQ(const LayerData &dataIn) const {
        computeNaive(dataIn);
    }

    // Compute the Dense using a tiled approach
    void DenseLayer::computeLogQ(const LayerData &dataIn) const {
        computeNaive(dataIn);
    }


    // Compute the Dense using SIMD
    void DenseLayer::computeHash(const LayerData &dataIn) const {
        computeNaive(dataIn);
    }
};