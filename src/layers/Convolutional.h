#pragma once

#include "Layer.h"
#include "../Utils.h"
#include "../Types.h"

namespace ML {
    class ConvolutionalLayer : public Layer {
        public:
            ConvolutionalLayer(const LayerParams inParams, const LayerParams outParams,
                               const LayerParams weightParams, const LayerParams biasParams, const fp32 weights_max, const fp32 inputs_max, const bool quantize, const int quant_length)
                : Layer(inParams, outParams, LayerType::CONVOLUTIONAL), weightParam(weightParams), weightData(weightParams),
                  biasParam(biasParams), biasData(biasParams), 
                  weightData_q(weightParams), biasData_q(biasParams), inputData_q(inParams), outputData_q(outParams),
                  weights_max(weights_max), inputs_max(inputs_max), quantize(quantize), quant_length(quant_length) {}

            // Getters
            const LayerParams& getWeightParams() const { return weightParam; }
            const LayerParams& getBiasParams() const { return biasParam; }
            const LayerData& getWeightData() const { return weightData; }
            const LayerData& getBiasData() const { return biasData; }
            const LayerData& getWeightData_q() const { return weightData_q; }
            const LayerData& getBiasData_q() const { return biasData_q; }
            const LayerData& getInputData_q() const { return inputData_q; }
            const LayerData& getOutputData_q() const { return outputData_q; }
            const fp32& getWeightsMax() const { return weights_max; }
            const fp32& getInputsMax() const { return inputs_max; }
            const bool& isQuantized() const { return quantize; }
            const int& getQuantLength() const { return quant_length; }

            // Allocate all resources needed for the layer & Load all of the required data for the layer
            template<typename T>
            void allocateLayer() {
                Layer::allocateOutputBuffer<Array3D<T>>();
                weightData.loadData<Array4D<T>>();
                biasData.loadData<Array1D<T>>();
                if(quantize) {
                    weightData_q.allocData<Array4D<T>>();
                    biasData_q.allocData<Array4D<T>>();
                    inputData_q.allocData<Array3D<T>>();
                    outputData_q.allocData<Array3D<T>>();
                }
            }

            // Free all resources allocated for the layer
            template<typename T>
            void freeLayer() {
                Layer::freeOutputBuffer<Array3D<T>>();
                weightData.freeData<Array4D<T>>();
                biasData.freeData<Array1D<T>>();

                if(quantize) {
                    weightData_q.freeData<Array4D<T>>();
                    biasData_q.freeData<Array4D<T>>();
                    inputData_q.freeData<Array3D<T>>();
                    outputData_q.freeData<Array3D<T>>();
                }
            }

            // Virtual functions
            virtual void computeNaive(const LayerData &dataIn) const override;
            virtual void computeThreaded(const LayerData &dataIn) const override;
            virtual void computeTiled(const LayerData &dataIn) const override;
            virtual void computeSIMD(const LayerData &dataIn) const override;

        private:
            LayerParams weightParam;
            LayerData weightData;
            LayerData weightData_q;

            LayerParams biasParam;
            LayerData biasData;
            LayerData biasData_q;

            LayerData inputData_q;
            LayerData outputData_q;
            
            fp32 weights_max;
            fp32 inputs_max;
            bool quantize;
            int quant_length;
    };

}