#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "Convolutional.h"
#include <chrono>
#include <cmath>
#include <algorithm>

using namespace std::chrono;
using namespace std;

extern float layer_times[12];
extern int layer_idx;

namespace ML {
    // --- Begin Student Code ---

    // Compute the convultion for the layer data
    void ConvolutionalLayer::computeNaive(const LayerData &dataIn) const {

        //Define Parameters
        int input_height = getInputParams().dims[0];
        int input_width = getInputParams().dims[1];
        int num_input_channels = getInputParams().dims[2];
        int filter_height = getWeightParams().dims[0];
        int filter_width = getWeightParams().dims[1];
        int num_filter_channels = getWeightParams().dims[3];
        int output_height, output_width;

        //Probably have a variable assicated with this        
        int batch_size = 1;
        int stride_size = 1;

        int input_x, input_y;

        output_height = ((input_height - filter_height + stride_size) / stride_size);
        output_width = ((input_width - filter_width + stride_size) / stride_size);

        //predeclair variables
        int n,m,p,q,c,r,s;

        //Preset LayerDate Tpye
        LayerData Weight_data = getWeightData();
        LayerData Bias_data = getBiasData();
        LayerData Output_data = getOutputData();

        //Map values to memory
        Array4D_fp32 convWeightData = Weight_data.getData<Array4D_fp32>();
        Array1D_fp32 convBiasData = Bias_data.getData<Array1D_fp32>();
        Array3D_fp32 convInputData = dataIn.getData<Array3D_fp32>();
        Array3D_fp32 convOutputData = Output_data.getData<Array3D_fp32>();

        //Start time for profiling
        auto start = high_resolution_clock::now();

        for(n = 0; n < batch_size; n++){
            for(p = 0; p < output_height; p++){
                for(q = 0; q < output_width; q++){
                    for(m = 0; m < num_filter_channels; m++) {
                        // std::cout << "\n" << p << q << m << "\n";
                        for(r = 0; r < filter_height; r++){
                            for(s = 0; s < filter_width; s++) { 
                                for(c = 0; c < num_input_channels; c++) {
                                    // std::cout << input_x << input_y << c << "\n";
                                    input_x = stride_size * q + s;
                                    input_y = stride_size * p + r;
                                    convOutputData[q][p][m] += convInputData[input_x][input_y][c] * convWeightData[s][r][c][m];
                                }
                            }
                        } 
                        convOutputData[q][p][m] += convBiasData[m];
                        if(convOutputData[q][p][m] < 0) { convOutputData[q][p][m] = 0; }
                    }
                } 
            }
        }

        auto end = high_resolution_clock::now();
        auto total = duration_cast<microseconds>(end - start);
        layer_times[layer_idx++] = total.count();
        // printf("Convolution Finished in %d us\n\r", total.count());
    }


    // Compute the convolution using threads
    void ConvolutionalLayer::computeLinearQ(const LayerData &dataIn) const {
        bool debug = false, debug_out = true;
        int quant_length = getQuantLength();

        //Define Parameters
        int input_height = getInputParams().dims[0];
        int input_width = getInputParams().dims[1];
        int num_input_channels = getInputParams().dims[2];
        int filter_height = getWeightParams().dims[0];
        int filter_width = getWeightParams().dims[1];
        int num_filter_channels = getWeightParams().dims[3];
        int output_height, output_width;

        //Probably have a variable assicated with this        
        int batch_size = 1;
        int stide_size = 1;

        output_height = ((input_height - filter_height + stide_size) / stide_size);
        output_width = ((input_width - filter_width + stide_size) / stide_size);

        //predeclair variables
        int n,m,p,q,c,r,s;
        int x,y,z,w;

        //Preset LayerDate Tpye
        LayerData Weight_data = getWeightData();
        LayerData Bias_data = getBiasData();
        LayerData Output_data = getOutputData();

        //Map values to memory
        Array4D_fp32 convWeightData = Weight_data.getData<Array4D_fp32>();
        Array1D_fp32 convBiasData = Bias_data.getData<Array1D_fp32>();
        Array3D_fp32 convInputData = dataIn.getData<Array3D_fp32>();
        Array3D_fp32 convOutputData = Output_data.getData<Array3D_fp32>();
        fp32 convOutputData_2[output_height][output_width][num_filter_channels];
            
        Array4D_i8 convWeightData_q = getWeightData_q().getData<Array4D_i8>();
        Array1D_i32 convBiasData_q = getBiasData_q().getData<Array1D_i32>();
        Array3D_ui8 convInputData_q = getInputData_q().getData<Array3D_ui8>();
        Array3D_i32 convOutputData_q = getOutputData_q().getData<Array3D_i32>();

        //Debugging Var - var[x][y][z]
        int input_x, input_y;

        //Quantize weights, inputs, and biases
        fp32 err = 0, err_rms = 0;
        fp32 minerr = 0, maxerr = 0;

        fp32 weights_max = convWeightData[0][0][0][0];
        fp32 inputs_max = convInputData[0][0][0];
        for(x = 0; x < getWeightParams().dims[0]; x++) {
            for(y = 0; y < getWeightParams().dims[1]; y++) {
                for(z = 0; z < getWeightParams().dims[2]; z++) {
                    for(w = 0; w < getWeightParams().dims[3]; w++) {
                        weights_max = std::max(weights_max, std::abs(convWeightData[x][y][z][w]));
                    }
                }
            }
        }
        for(x = 0; x < getInputParams().dims[0]; x++) {
            for(y = 0; y < getInputParams().dims[1]; y++) {
                for(z = 0; z < getInputParams().dims[2]; z++) {
                    inputs_max = std::max(inputs_max, std::abs(convInputData[x][y][z]));
                }
            }
        }

        //Create Scales for quantization
        i8 scale_weight = std::pow(2,quant_length-1)-1 / weights_max;
        ui8 scale_input = std::pow(2,quant_length)-1 / inputs_max;
        i32 scale_biases = scale_input * scale_weight;
        // printf("scale_weight: %d\n\rscale_input: %d\n\rscale_biases: %d\n\r", scale_weight, scale_input, scale_biases);

        if(debug || debug_out)  {
            std::cout << "------------------------layer_" << layer_idx << "------------------------\n";
        }

        for(x = 0; x < getWeightParams().dims[0]; x++) {
            for(y = 0; y < getWeightParams().dims[1]; y++) {
                for(z = 0; z < getWeightParams().dims[2]; z++) {
                    for(w = 0; w < getWeightParams().dims[3]; w++) {
                        //i8 w_q = std::log2(convWeightData[x][y][z][w]);
                        fp32 weight = convWeightData[x][y][z][w];
                        i8 weight_q = static_cast<i8>(weight * scale_weight);

                        convWeightData_q[x][y][z][w] = weight_q;
                        
                        if(debug) {
                            fp32 w_dq = weight_q / static_cast<fp32>(scale_weight);

                            // std::cout << "weight_fp32: " << convWeightData[x][y][z][w] << "\n"
                            //           << "weight_log2: " << (int)weight_q << "\n" 
                            //           << "weight_deqt: " << w_dq << "\n\n";

                            fp32 cerr = std::fabs(weight - w_dq);
                            err += cerr;
                            minerr = std::min(minerr, cerr);
                            maxerr = std::max(maxerr, cerr);

                            cerr = std::pow(weight - w_dq, 2);
                            err_rms += cerr;
                        }
                    }
                }
            }
        }

        if(debug) { 
            std::cout << "-----------weights-------------\n";
            std::cout << "avg err: " << err/(getWeightParams().dims[0] * getWeightParams().dims[1] * getWeightParams().dims[2] * getWeightParams().dims[3]) << "\n"
                      << "rms err: " << std::sqrt(err_rms/(getWeightParams().dims[0] * getWeightParams().dims[1] * getWeightParams().dims[2] * getWeightParams().dims[3])) << "\n"
                      << "min err: " << minerr << "\n"
                      << "max err: " << maxerr << "\n\n";
        }

        err = 0, err_rms = 0, minerr = 0, maxerr = 0;
        for(x = 0; x < getInputParams().dims[0]; x++) {
            for(y = 0; y < getInputParams().dims[1]; y++) {
                for(z = 0; z < getInputParams().dims[2]; z++) {
                    fp32 input = convInputData[x][y][z];
                    ui8 input_q = static_cast<ui8>(input * scale_input);

                    convInputData_q[x][y][z] = input_q;

                    if(debug) {
                        fp32 i_dq = input_q / static_cast<fp32>(scale_input);

                        // std::cout << "input_fp32: " << input << "\n"
                        //           << "input_log2: " << (int)input_q << "\n" 
                        //           << "input_deqt: " << i_dq << "\n\n";

                        fp32 cerr = std::fabs(input - i_dq);
                        err += cerr;
                        minerr = std::min(minerr, cerr);
                        maxerr = std::max(maxerr, cerr);

                        cerr = std::pow(input - i_dq, 2);
                        err_rms += cerr;
                    }
                }
            }
        }

        if(debug) { 
            std::cout << "-----------inputs-------------\n";
            std::cout << "avg err: " << err/(getInputParams().dims[0] * getInputParams().dims[1] * getInputParams().dims[2]) << "\n"
                      << "rms err: " << std::sqrt(err_rms/(getInputParams().dims[0] * getInputParams().dims[1] * getInputParams().dims[2])) << "\n"
                      << "min err: " << minerr << "\n"
                      << "max err: " << maxerr << "\n\n";
        }

        err = 0, err_rms = 0, minerr = 0, maxerr = 0;
        for(x = 0; x < getBiasParams().dims[0]; x++) {
            fp32 bias = convBiasData[x];
            i32 bias_q = static_cast<i32>(bias * scale_biases);

            convBiasData_q[x] = bias_q;
            
            if(debug) {
                fp32 b_dq = bias_q / static_cast<fp32>(scale_biases);

                // std::cout << "bias_fp32: " << bias << "\n"
                //           << "bias_log2: " << (int)bias_q << "\n" 
                //           << "bias_deqt: " << b_dq << "\n\n";

                fp32 cerr = std::fabs(bias - b_dq);
                err += cerr;
                minerr = std::min(minerr, cerr);
                maxerr = std::max(maxerr, cerr);

                cerr = std::pow(bias - b_dq, 2);
                err_rms += cerr;
            }
        }

        if(debug) { 
            std::cout << "-----------biases-------------\n";
            std::cout << "avg err: " << err/(getBiasParams().dims[0]) << "\n"
                      << "rms err: " << std::sqrt(err_rms/(getBiasParams().dims[0])) << "\n"
                      << "min err: " << minerr << "\n"
                      << "max err: " << maxerr << "\n\n";
        }

        for(x = 0; x < getOutputParams().dims[0]; x++) {
            for(y = 0; y < getOutputParams().dims[1]; y++) {
                for(z = 0; z < getOutputParams().dims[2]; z++) { 
                    convOutputData_2[x][y][z] = 0;
                }
            }
        }

        //Start time for profiling
        auto start = high_resolution_clock::now();

        for(n = 0; n < batch_size; n++){
            for(p = 0; p < output_height; p++){
                for(q = 0; q < output_width; q++){
                    for(m = 0; m < num_filter_channels; m++) {
                        for(r = 0; r < filter_height; r++){
                            for(s = 0; s < filter_width; s++) { 
                                for(c = 0; c < num_input_channels; c++) {
                                    input_x = stide_size * q + s;
                                    input_y = stide_size * p + r;
                                    convOutputData_q[q][p][m] += convInputData_q[input_x][input_y][c] * convWeightData_q[s][r][c][m];
                                }
                            }
                        } 
                        convOutputData_q[q][p][m] += convBiasData_q[m];
                    }
                } 
            }
        }

        auto end = high_resolution_clock::now();
        auto total = duration_cast<microseconds>(end - start);
        layer_times[layer_idx++] = total.count();

        if(debug_out) {
            for(n = 0; n < batch_size; n++){
                for(p = 0; p < output_height; p++){
                    for(q = 0; q < output_width; q++){
                        for(m = 0; m < num_filter_channels; m++) {
                            for(r = 0; r < filter_height; r++){
                                for(s = 0; s < filter_width; s++) { 
                                    for(c = 0; c < num_input_channels; c++) {
                                        input_x = stide_size * q + s;
                                        input_y = stide_size * p + r;
                                        convOutputData_2[q][p][m] += convInputData[input_x][input_y][c] * convWeightData[s][r][c][m];
                                        if(std::isnan(convOutputData_2[q][p][m])) {

                                        }
                                    }
                                }
                            } 
                            convOutputData_2[q][p][m] += convBiasData[m];
                            if(convOutputData_2[q][p][m] < 0) { convOutputData_2[q][p][m] = 0; }
                        }
                    } 
                }
            }
        }

        //De-quantize output values back to fp32
        err = 0, err_rms = 0, minerr = 0, maxerr = 0;
        for(x = 0; x < getOutputParams().dims[0]; x++) {
            for(y = 0; y < getOutputParams().dims[1]; y++) {
                for(z = 0; z < getOutputParams().dims[2]; z++) {
                    fp32 o_dq;
                    fp32 output = convOutputData_2[x][y][z];
                    i32 output_q = convOutputData_q[x][y][z];

                    if(output_q > 0) {
                        o_dq = output_q / static_cast<fp32>(scale_biases);
                    } else {
                        o_dq = 0;
                    }

                    convOutputData[x][y][z] = o_dq;

                    std::cout << "output_fp32: " << output << "\n"
                              << "output_log2: " << (int)output_q << "\n" 
                              << "output_deqt: " << o_dq << "\n\n";

                    if(debug_out) {
                        fp32 cerr = std::fabs(output - o_dq);
                        err += cerr;
                        minerr = std::min(minerr, cerr);
                        maxerr = std::max(maxerr, cerr);

                        cerr = std::pow(output - o_dq, 2);
                        err_rms += cerr;
                    }
                }
            }
        }

        if(debug_out) { 
            std::cout << "-----------outputs-------------\n";
            std::cout << "avg err: " << err/(getOutputParams().dims[0] * getOutputParams().dims[1] * getOutputParams().dims[2]) << "\n"
                      << "rms err: " << std::sqrt(err_rms/(getOutputParams().dims[0] * getOutputParams().dims[1] * getOutputParams().dims[2])) << "\n"
                      << "min err: " << minerr << "\n"
                      << "max err: " << maxerr << "\n"
                      << "-------------------------------------------------------\n\n";
        }
    }


    // Compute the convolution using non-linear (log) scaling
    void ConvolutionalLayer::computeLogQ(const LayerData &dataIn) const {
        bool debug = false, debug_out = true;

        //Define Parameters
        int input_height = getInputParams().dims[0];
        int input_width = getInputParams().dims[1];
        int num_input_channels = getInputParams().dims[2];
        int filter_height = getWeightParams().dims[0];
        int filter_width = getWeightParams().dims[1];
        int num_filter_channels = getWeightParams().dims[3];
        int output_height, output_width;

        //Probably have a variable assicated with this        
        int batch_size = 1;
        int stide_size = 1;

        output_height = ((input_height - filter_height + stide_size) / stide_size);
        output_width = ((input_width - filter_width + stide_size) / stide_size);

        //predeclair variables
        int n,m,p,q,c,r,s;
        int x,y,z,w;

        //Preset LayerDate Tpye
        LayerData Weight_data = getWeightData();
        LayerData Bias_data = getBiasData();
        LayerData Output_data = getOutputData();

        //Map values to memory
        Array4D_fp32 convWeightData = Weight_data.getData<Array4D_fp32>();
        Array1D_fp32 convBiasData = Bias_data.getData<Array1D_fp32>();
        Array3D_fp32 convInputData = dataIn.getData<Array3D_fp32>();
        Array3D_fp32 convOutputData = Output_data.getData<Array3D_fp32>();
        fp32 convOutputData_2[output_height][output_width][num_filter_channels];
            
        Array4D_i8 convWeightData_q = getWeightData_q().getData<Array4D_i8>();
        Array1D_i8 convBiasData_q = getBiasData_q().getData<Array1D_i8>();
        Array3D_ui8 convInputData_q = getInputData_q().getData<Array3D_ui8>();
        Array3D_i8 convOutputData_q = getOutputData_q().getData<Array3D_i8>();

        //Debugging Var - var[x][y][z]
        int input_x, input_y;

        //Quantize weights, inputs, and biases
        fp32 err = 0, err_rms = 0;
        fp32 minerr = 0, maxerr = 0;

        if(debug || debug_out)  {
            std::cout << "------------------------layer_" << layer_idx << "------------------------\n";
        }

        int ceil = 128;
        for(x = 0; x < getWeightParams().dims[0]; x++) {
            for(y = 0; y < getWeightParams().dims[1]; y++) {
                for(z = 0; z < getWeightParams().dims[2]; z++) {
                    for(w = 0; w < getWeightParams().dims[3]; w++) {
                        //i8 w_q = std::log2(convWeightData[x][y][z][w]);
                        fp32 weight = convWeightData[x][y][z][w];
                        i8 weight_q;
                        if(weight < 0) {
                            weight_q = static_cast<i8>(std::clamp(-(std::round(std::log2(std::fabs(weight)))), (float)1.0, (float)ceil-1));
                            weight_q = -weight_q;
                        } else if(weight > 0) {
                            weight_q = static_cast<i8>(std::clamp(-(std::round(std::log2(std::fabs(weight)))), (float)1.0, (float)ceil));
                        } else {
                            weight_q = 0;
                        }

                        convWeightData_q[x][y][z][w] = weight_q;
                        
                        if(debug) {
                            fp32 w_dq;
                            if(weight_q > 0) {
                                w_dq = (std::pow(2, -(std::abs(weight_q))));
                            } else if(weight_q < 0) {
                                w_dq = (-(std::pow(2, -(std::abs(weight_q)))));
                            } else {
                                w_dq = 0;
                            }

                            // std::cout << "weight_fp32: " << convWeightData[x][y][z][w] << "\n"
                            //           << "weight_log2: " << (int)weight_q << "\n" 
                            //           << "weight_deqt: " << w_dq << "\n\n";
                            fp32 cerr = std::fabs(convWeightData[x][y][z][w] - w_dq);
                            err += cerr;
                            minerr = std::min(minerr, cerr);
                            maxerr = std::max(maxerr, cerr);

                            cerr = std::pow(convWeightData[x][y][z][w] - w_dq, 2);
                            err_rms += cerr;
                        }
                    }
                }
            }
        }

        if(debug) { 
            std::cout << "-----------weights-------------\n";
            std::cout << "avg err: " << err/(getWeightParams().dims[0] * getWeightParams().dims[1] * getWeightParams().dims[2] * getWeightParams().dims[3]) << "\n"
                      << "rms err: " << std::sqrt(err_rms/(getWeightParams().dims[0] * getWeightParams().dims[1] * getWeightParams().dims[2] * getWeightParams().dims[3])) << "\n"
                      << "min err: " << minerr << "\n"
                      << "max err: " << maxerr << "\n\n";
        }

        err = 0, err_rms = 0, minerr = 0, maxerr = 0;
        ceil = 256;
        for(x = 0; x < getInputParams().dims[0]; x++) {
            for(y = 0; y < getInputParams().dims[1]; y++) {
                for(z = 0; z < getInputParams().dims[2]; z++) {
                    fp32 input = convInputData[x][y][z];
                    ui8 input_q;
                    if(input > 0) {
                        input_q = static_cast<ui8>(std::clamp(-(std::round(std::log2(input))), (float)1.0, (float)ceil-1));
                    } else {
                        input_q = 0;
                    }

                    convInputData_q[x][y][z] = input_q;

                    if(debug) {
                        fp32 i_dq;
                        if(input_q > 0) {
                            i_dq = (std::pow(2, -(std::abs(input_q))));
                        } else {
                            i_dq = 0;
                        }
                        
                        // if(layer_idx > 0) {
                        //     std::cout << "input_fp32: " << input << "\n"
                        //               << "input_log2: " << (int)input_q << "\n" 
                        //               << "input_deqt: " << i_dq << "\n\n";
                        // }

                        fp32 cerr = std::fabs(input - i_dq);
                        err += cerr;
                        minerr = std::min(minerr, cerr);
                        maxerr = std::max(maxerr, cerr);

                        cerr = std::pow(input - i_dq, 2);
                        err_rms += cerr;
                    }
                }
            }
        }

        if(debug) { 
            std::cout << "-----------inputs-------------\n";
            std::cout << "avg err: " << err/(getInputParams().dims[0] * getInputParams().dims[1] * getInputParams().dims[2]) << "\n"
                      << "rms err: " << std::sqrt(err_rms/(getInputParams().dims[0] * getInputParams().dims[1] * getInputParams().dims[2])) << "\n"
                      << "min err: " << minerr << "\n"
                      << "max err: " << maxerr << "\n\n";
        }

        err = 0, err_rms = 0, minerr = 0, maxerr = 0;
        ceil = 128;
        for(x = 0; x < getBiasParams().dims[0]; x++) {
            fp32 bias = convBiasData[x];
            i8 bias_q;
            if(bias < 0) {
                bias_q = static_cast<i8>(std::clamp(-(std::round(std::log2(std::fabs(bias)))), (float)1.0, (float)ceil-1));
                bias_q = -bias_q;
            } else if(bias > 0) {
                bias_q = static_cast<i8>(std::clamp(-((std::round(std::log2(std::fabs(bias))))), (float)1.0, (float)ceil));
            } else {
                bias_q = 0;
            }

            convBiasData_q[x] = bias_q;
            
            if(debug) {
                fp32 b_dq;
                if(bias_q > 0) {
                    b_dq = (std::pow(2, -(std::abs(bias_q))));
                } else if(bias_q < 0) {
                    b_dq = -(std::pow(2, -(std::abs(bias_q))));
                } else {
                    b_dq = 0;
                }

                // std::cout << "bias_fp32: " << bias << "\n"
                //           << "bias_log2: " << (int)bias_q << "\n" 
                //           << "bias_deqt: " << b_dq << "\n\n";

                fp32 cerr = std::fabs(bias - b_dq);
                err += cerr;
                minerr = std::min(minerr, cerr);
                maxerr = std::max(maxerr, cerr);

                cerr = std::pow(bias - b_dq, 2);
                err_rms += cerr;
            }
        }

        if(debug) { 
            std::cout << "-----------biases-------------\n";
            std::cout << "avg err: " << err/(getBiasParams().dims[0]) << "\n"
                      << "rms err: " << std::sqrt(err_rms/(getBiasParams().dims[0])) << "\n"
                      << "min err: " << minerr << "\n"
                      << "max err: " << maxerr << "\n\n";
        }

        for(x = 0; x < getOutputParams().dims[0]; x++) {
            for(y = 0; y < getOutputParams().dims[1]; y++) {
                for(z = 0; z < getOutputParams().dims[2]; z++) { 
                    convOutputData_2[x][y][z] = 0;
                    convOutputData_q[x][y][z] = 0;
                }
            }
        }

        //Start time for profiling
        i8 prev_acc = 0;

        auto start = high_resolution_clock::now();

        for(n = 0; n < batch_size; n++){
            for(p = 0; p < output_height; p++){
                for(q = 0; q < output_width; q++){
                    for(m = 0; m < num_filter_channels; m++) {
                        for(r = 0; r < filter_height; r++){
                            for(s = 0; s < filter_width; s++) { 
                                for(c = 0; c < num_input_channels; c++) {
                                    input_x = stide_size * q + s;
                                    input_y = stide_size * p + r;
                                    // convOutputData_q[q][p][m] += convInputData_q[input_x][input_y][c] * convWeightData_q[s][r][c][m];

                                    // MAC in Log Domain is a bit convoluted (lol)
                                    // pn = inputn + weightn == log2(inputn * weightn)
                                    // sn-1 = previous accumulation
                                    // sn = current accumulation
                                    i8 pn = convInputData_q[input_x][input_y][c] + convWeightData_q[s][r][c][m];
                                    convOutputData_q[q][p][m] = std::max(prev_acc, pn) + (1 << -(static_cast<i8>((std::abs(prev_acc - pn)))));
                                    prev_acc = convOutputData_q[q][p][m];

                                    // std::cout << (int)convOutputData[q][p][m] << "\n";
                                }
                            }
                        } 
                        // Accumulate bias 
                        i8 bias = convBiasData_q[m];
                        convOutputData_q[q][p][m] = std::max(prev_acc, bias) + (1 << -(static_cast<i8>((std::abs(prev_acc - bias)))));
                        prev_acc = 0;
                    }
                } 
            }
        }

        auto end = high_resolution_clock::now();
        auto total = duration_cast<microseconds>(end - start);
        layer_times[layer_idx++] = total.count();

        if(debug_out) {
            for(n = 0; n < batch_size; n++){
                for(p = 0; p < output_height; p++){
                    for(q = 0; q < output_width; q++){
                        for(m = 0; m < num_filter_channels; m++) {
                            for(r = 0; r < filter_height; r++){
                                for(s = 0; s < filter_width; s++) { 
                                    for(c = 0; c < num_input_channels; c++) {
                                        input_x = stide_size * q + s;
                                        input_y = stide_size * p + r;
                                        convOutputData_2[q][p][m] += convInputData[input_x][input_y][c] * convWeightData[s][r][c][m];
                                        if(std::isnan(convOutputData_2[p][q][m])) {

                                        }
                                    }
                                }
                            } 
                            convOutputData_2[q][p][m] += convBiasData[m];
                            if(convOutputData_2[q][p][m] < 0) { convOutputData_2[q][p][m] = 0; }
                        }
                    } 
                }
            }
        }

        //De-quantize output values back to fp32
        err = 0, err_rms = 0, minerr = 0, maxerr = 0;
        for(x = 0; x < getOutputParams().dims[0]; x++) {
            for(y = 0; y < getOutputParams().dims[1]; y++) {
                for(z = 0; z < getOutputParams().dims[2]; z++) {
                    fp32 o_dq;
                    fp32 output = convOutputData_2[x][y][z];
                    i32 output_q = convOutputData_q[x][y][z];

                    if(output_q > 0) {
                        o_dq = static_cast<fp32>(std::pow(2, -(std::abs(output_q))));
                    } else {
                        o_dq = 0;
                    }

                    convOutputData[x][y][z] = o_dq;

                    std::cout << "output_fp32: " << output << "\n"
                              << "output_log2: " << (int)output_q << "\n" 
                              << "output_deqt: " << o_dq << "\n\n";

                    if(debug_out) {
                        fp32 cerr = std::fabs(output - o_dq);
                        err += cerr;
                        minerr = std::min(minerr, cerr);
                        maxerr = std::max(maxerr, cerr);

                        cerr = std::pow(output - o_dq, 2);
                        err_rms += cerr;
                    }
                }
            }
        }

        if(debug_out) { 
            std::cout << "-----------outputs-------------\n";
            std::cout << "avg err: " << err/(getOutputParams().dims[0] * getOutputParams().dims[1] * getOutputParams().dims[2]) << "\n"
                      << "rms err: " << std::sqrt(err_rms/(getOutputParams().dims[0] * getOutputParams().dims[1] * getOutputParams().dims[2])) << "\n"
                      << "min err: " << minerr << "\n"
                      << "max err: " << maxerr << "\n"
                      << "-------------------------------------------------------\n\n";
        }
    }


    // Compute the convolution using SIMD
    void ConvolutionalLayer::computeHash(const LayerData &dataIn) const {
        // TODO: Your Code Here...

        std::cout << "\n\n\nhello Convolution SIMD\n\n\n";

    }
};