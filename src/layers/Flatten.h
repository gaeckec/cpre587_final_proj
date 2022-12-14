#pragma once

#include "Layer.h"
#include "../Utils.h"
#include "../Types.h"

namespace ML {
    class FlattenLayer : public Layer {
        public:
            FlattenLayer(const LayerParams inParams, const LayerParams outParams)
                : Layer(inParams, outParams, LayerType::FLATTEN) {}

            // Getters
            // const LayerParams& getWeightParams() const { return weightParam; }
            // const LayerParams& getBiasParams() const { return biasParam; }
            // const LayerData& getWeightData() const { return weightData; }
            // const LayerData& getBiasData() const { return biasData; }

            // Allocate all resources needed for the layer & Load all of the required data for the layer
            template<typename T>
            void allocateLayer() {
                Layer::allocateOutputBuffer<Array3D<T>>();
                // weightData.loadData<Array4D<T>>();
                // biasData.loadData<Array1D<T>>();
            }

            // Free all resources allocated for the layer
            template<typename T>
            void freeLayer() {
                Layer::freeOutputBuffer<Array3D<T>>();
                // weightData.freeData<Array4D<T>>();
                // biasData.freeData<Array1D<T>>();
            }

            // Virtual functions
            virtual void computeNaive(const LayerData &dataIn) const override;
            virtual void computeLinearQ(const LayerData &dataIn) const override;
            virtual void computeLogQ(const LayerData &dataIn) const override;
            virtual void computeHash(const LayerData &dataIn) const override;

        private:
            // LayerParams weightParam;
            // LayerData weightData;

            // LayerParams biasParam;
            // LayerData biasData;
    };

}