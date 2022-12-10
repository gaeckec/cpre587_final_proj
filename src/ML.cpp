#include <iostream>
#include <filesystem>
#include <vector>

#include "Utils.h"
#include "Types.h"
#include "Model.h"
#include "layers/Layer.h"
#include "layers/Convolutional.h"
#include "layers/Dense.h"
#include "layers/MaxPooling.h"
#include "layers/Softmax.h"
#include "layers/Flatten.h"

#define need_4_speed 0


// Make our code a bit cleaner
namespace fs = std::filesystem;
using namespace ML;

float layer_times[12];
int layer_idx;

// Build our ML toy model
Model buildToyModel(const fs::path modelPath) {
    Model model;
    bool quantize = true;
    std::cout << "\n--- Building Toy Model ---" << std::endl;

    // --- Conv 1: L0 ---
    // Input shape: 64x64x3
    // Output shape: 60x60x32
    // LayerParams conv1_inDataParam = getParams<fp32, "", 64, 64, 3>();
    // LayerParams conv1_outDataParam = getParams<fp32, "", 60, 60, 32>();
    // LayerParams conv1_weightParam = getParams<fp32, "", 5, 5, 3, 32>();
    // LayerParams conv1_biasParam = getParams<fp32, "", 32>();
    LayerParams conv1_inDataParam(sizeof(fp32), {64, 64, 3});
    LayerParams conv1_outDataParam(sizeof(fp32), {60, 60, 32});
    LayerParams conv1_weightParam(sizeof(fp32), {5, 5, 3, 32}, modelPath / "conv1_weights.bin");
    LayerParams conv1_biasParam(sizeof(fp32), {32}, modelPath / "conv1_biases.bin");

    ConvolutionalLayer* conv1 = new ConvolutionalLayer(conv1_inDataParam, conv1_outDataParam, conv1_weightParam, conv1_biasParam, 8);
    model.addLayer(conv1);

    // --- Conv 2: L1 ---
    // Input shape: 60x60x32
    // Output shape: 56x56x32
    LayerParams conv2_inDataParam(sizeof(fp32), {60, 60, 32});
    LayerParams conv2_outDataParam(sizeof(fp32), {56, 56, 32});
    LayerParams conv2_weightParam(sizeof(fp32), {5, 5, 32, 32}, modelPath / "conv2_weights.bin");
    LayerParams conv2_biasParam(sizeof(fp32), {32}, modelPath / "conv2_biases.bin");

    ConvolutionalLayer* conv2 = new ConvolutionalLayer(conv2_inDataParam, conv2_outDataParam, conv2_weightParam, conv2_biasParam, 8);
    model.addLayer(conv2);

    // --- MPL 0: L2 ---
    // Input shape: 56x56x32
    // Output shape: 28x28x32
    LayerParams mpl0_inDataParam(sizeof(fp32), {56, 56, 32});
    LayerParams mpl0_outDataParam(sizeof(fp32), {28, 28, 32});
    
    MaxPoolingLayer* mpl0 = new MaxPoolingLayer(mpl0_inDataParam, mpl0_outDataParam);
    model.addLayer(mpl0);

    // --- Conv 3: L3 ---
    // Input shape: 28x28x32
    // Output shape: 26x26x64
    LayerParams conv3_inDataParam(sizeof(fp32), {28, 28, 32});
    LayerParams conv3_outDataParam(sizeof(fp32), {26, 26, 64});
    LayerParams conv3_weightParam(sizeof(fp32), {3, 3, 32, 64}, modelPath / "conv3_weights.bin");
    LayerParams conv3_biasParam(sizeof(fp32), {64}, modelPath / "conv3_biases.bin");

    ConvolutionalLayer* conv3 = new ConvolutionalLayer(conv3_inDataParam, conv3_outDataParam, conv3_weightParam, conv3_biasParam, 8);
    model.addLayer(conv3);

    // --- Conv 4: L4 ---
    // Input shape: 26x26x64
    // Output shape: 24x24x64
    LayerParams conv4_inDataParam(sizeof(fp32), {26, 26, 64});
    LayerParams conv4_outDataParam(sizeof(fp32), {24, 24, 64});
    LayerParams conv4_weightParam(sizeof(fp32), {3, 3, 64, 64}, modelPath / "conv4_weights.bin");
    LayerParams conv4_biasParam(sizeof(fp32), {64}, modelPath / "conv4_biases.bin");

    ConvolutionalLayer* conv4 = new ConvolutionalLayer(conv4_inDataParam, conv4_outDataParam, conv4_weightParam, conv4_biasParam, 8);
    model.addLayer(conv4);

    // --- MPL 1: L5---
    // Input shape: 24x24x64
    // Output shape: 12x12x64
    LayerParams mpl1_inDataParam(sizeof(fp32), {24, 24, 64});
    LayerParams mpl1_outDataParam(sizeof(fp32), {12, 12, 64});

    MaxPoolingLayer* mpl1 = new MaxPoolingLayer(mpl1_inDataParam, mpl1_outDataParam);
    model.addLayer(mpl1);

    // --- Conv 5: L6 ---
    // Input shape: 12x12x64
    // Output shape: 10x10x64
    LayerParams conv5_inDataParam(sizeof(fp32), {12, 12, 64});
    LayerParams conv5_outDataParam(sizeof(fp32), {10, 10, 64});
    LayerParams conv5_weightParam(sizeof(fp32), {3, 3, 64, 64}, modelPath / "conv5_weights.bin");
    LayerParams conv5_biasParam(sizeof(fp32), {64}, modelPath / "conv5_biases.bin");

    ConvolutionalLayer* conv5 = new ConvolutionalLayer(conv5_inDataParam, conv5_outDataParam, conv5_weightParam, conv5_biasParam, 8);
    model.addLayer(conv5);

    // --- Conv 6: L7 ---
    // Input shape: 10x10x64
    // Output shape: 8x8x128
    LayerParams conv6_inDataParam(sizeof(fp32), {10, 10, 64});
    LayerParams conv6_outDataParam(sizeof(fp32), {8, 8, 128});
    LayerParams conv6_weightParam(sizeof(fp32), {3, 3, 64, 128}, modelPath / "conv6_weights.bin");
    LayerParams conv6_biasParam(sizeof(fp32), {128}, modelPath / "conv6_biases.bin");

    ConvolutionalLayer* conv6 = new ConvolutionalLayer(conv6_inDataParam, conv6_outDataParam, conv6_weightParam, conv6_biasParam, 8);
    model.addLayer(conv6);

    // --- MPL 2: L8 ---
    // Input shape: 8x8x128
    // Output shape: 4x4x128
    LayerParams mpl2_inDataParam(sizeof(fp32), {8, 8, 128});
    LayerParams mpl2_outDataParam(sizeof(fp32), {4, 4, 128});

    MaxPoolingLayer* mpl2 = new MaxPoolingLayer(mpl2_inDataParam, mpl2_outDataParam);
    model.addLayer(mpl2);

    // --- Flatten 0: L9 ---
    // Input shape: 4x4x128
    // Output shape: 2048
    LayerParams flat0_inDataParam(sizeof(fp32), {4, 4, 128});
    LayerParams flat0_outDataParam(sizeof(fp32), {2048});

    FlattenLayer* flat0 = new FlattenLayer(flat0_inDataParam, flat0_outDataParam);
    model.addLayer(flat0);
/*
    // --- Dense 0: L10 ---
    // Input shape: 2048
    // Output shape: 256
    LayerParams dense1_inDataParam(sizeof(fp32), {2048});
    LayerParams dense1_outDataParam(sizeof(fp32), {256});
    LayerParams dense1_weightParam(sizeof(fp32), {2048, 256}, modelPath / "dense1_weights.bin");
    LayerParams dense1_biasParam(sizeof(fp32), {256}, modelPath / "dense1_biases.bin");

    DenseLayer* dense1 = new DenseLayer(dense1_inDataParam, dense1_outDataParam, dense1_weightParam, dense1_biasParam);
    model.addLayer(dense1);

    // --- Dense 1: L11 ---
    // Input shape: 256
    // Output shape: 200
    LayerParams dense2_inDataParam(sizeof(fp32), {256});
    LayerParams dense2_outDataParam(sizeof(fp32), {200});
    LayerParams dense2_weightParam(sizeof(fp32), {256, 200}, modelPath / "dense2_weights.bin");
    LayerParams dense2_biasParam(sizeof(fp32), {200}, modelPath / "dense2_biases.bin");

    DenseLayer* dense2 = new DenseLayer(dense2_inDataParam, dense2_outDataParam, dense2_weightParam, dense2_biasParam);
    model.addLayer(dense2);

    // --- Softmax 0: L12 ---
    // Input shape: 200
    // Output shape: 200
    LayerParams sm0_inDataParam(sizeof(fp32), {200});
    LayerParams sm0_outDataParam(sizeof(fp32), {200});

    SoftMaxLayer* sm0 = new SoftMaxLayer(sm0_inDataParam, sm0_outDataParam);
    model.addLayer(sm0);
*/
    return model;
}


void runTests() {
    std::cout << "\n--- Running Some Tests ---" << std::endl;

    // Load an image
    fs::path imgPath("./data/img_val_dataset/test_input_0.bin");
    dimVec dims = {64, 64, 3};
    Array3D_fp32 img = loadArray<Array3D_fp32>(imgPath, dims);

    // Compare images
    std::cout << "Comparing image 0 to itself (max error): "
              << compareArray<Array3D_fp32>(img, img, dims)
              << std::endl
              << "Comparing image 0 to itself (T/F within epsilon " << EPSILON << "): "
              << std::boolalpha
              << compareArrayWithin<Array3D_fp32>(img, img, dims, EPSILON)
              << std::endl;

    // Test again with a modified copy
    std::cout << "\nChange a value by 1.0 and compare again" << std::endl;
    Array3D_fp32 imgCopy = allocArray<Array3D_fp32>(dims);
    copyArray<Array3D_fp32>(img, imgCopy, dims);
    imgCopy[0][0][0] += 1.0;

    std::cout << "Comparing image 0 to itself (max error): "
              << compareArray<Array3D_fp32>(img, imgCopy, dims)
              << std::endl
              << "Comparing image 0 to itself (T/F within epsilon " << EPSILON << "): "
              << std::boolalpha
              << compareArrayWithin<Array3D_fp32>(img, imgCopy, dims, EPSILON)
              << std::endl;


    // Clean up after ourselves
    freeArray<Array3D_fp32>(img, dims);
    freeArray<Array3D_fp32>(imgCopy, dims);
}


#ifdef ZEDBOARD
int runModelTest() {
#else
int main(int argc, char **argv) {
    // Hanlde command line arguments
    Args& args = Args::getInst();
    args.parseArgs(argc, argv);
#endif

    // Run some framework tests as an example of loading data
    runTests();

    // Base input data path (determined from current directory of where you are running the command)
//#ifdef ZEDBOARD
    fs::path basePath("data");
// #else
//     fs::path basePath("data");
// #endif

    // Build the model and allocate the buffers
    Model model = buildToyModel(basePath / "model");
    model.allocLayers<fp32>();

    // Load an image
    std::cout << "\n--- Running Infrence ---" << std::endl;
    dimVec dims = {64, 64, 3};

#if need_4_speed
    char str_filename[20];
    int times_index,value;
    int loops = 20;
    float time_count = 0;
    float min_latency = __INT64_MAX__;
    float max_latency = 0;
    float total_f = 0;
    // srand (time(NULL));
    srand (12102022);
    for (times_index = 0; times_index < loops; times_index++){
        // Construct a LayerData object from a LayerParams one
        value = rand() % 30000;
        sprintf(str_filename,"test_input_%d.bin", value);
        LayerData img( {sizeof(fp32), dims, basePath / "img_train_dataset" / str_filename} );
        printf("%2d: ", times_index);
        img.loadData<Array3D_fp32>();
        // auto start = std::chrono::high_resolution_clock::now();
        layer_idx = 0;
        // layer_times = {0};
        // Run infrence on the model
        const LayerData output = model.infrence(img, Layer::InfType::NAIVE);

        //Finishing timing of system
        // auto end = std::chrono::high_resolution_clock::now();
        // auto total = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_f = 0;
        for(int i = 0; i < 12; i++) {
            total_f += layer_times[i];
        } 

        if(total_f > max_latency) { max_latency = total_f; }
        if(total_f < min_latency) { min_latency = total_f; }

        time_count += total_f;

    }
    std::cout << "\n\nAverage output time =  " 
              << time_count / loops
              << " us \n"
              << "Min latency = "
              << min_latency
              << " us \n"
              << "Max latency = "
              << max_latency
              << " us \n";

#else
    // Construct a LayerData object from a LayerParams one
    LayerData img( {sizeof(fp32), dims, basePath / "image_0.bin"} );
    img.loadData<Array3D_fp32>();

    // auto start = std::chrono::high_resolution_clock::now();

    // Run infrence on the model
    const LayerData output = model.infrence(img, Layer::InfType::LINQ);

    //Finishing timing of system
    float latency = 0;
    for(int l = 0; l < 12; l++) {
        latency += layer_times[l];
    }

    std::cout << "Inference Latency: " << latency << " us\n";

    // Compare the output
    std::cout << "\n--- Comparing The Output ---" << std::endl;

    // Construct a LayerData object from a LayerParams one
    LayerData expected( { sizeof(fp32), {2048}, basePath / "image_0_data" / "layer_9_output.bin" } );
    expected.loadData<Array3D_fp32>();
    std::cout << "Comparing expected output to model output (max error / T/F within epsilon " << EPSILON << "): \n\t"
              << expected.compare<Array3D<fp32>>(output) << " / "
              << std::boolalpha << bool(expected.compareWithin<Array3D<fp32>>(output, EPSILON))
              << std::endl;

#endif

    // Clean up
    //model.freeLayers<fp32>();
    return 0;
}


#ifdef ZEDBOARD
    }; // namespace ML;
#endif