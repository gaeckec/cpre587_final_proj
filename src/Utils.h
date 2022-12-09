#pragma once

#include <filesystem>
#include <string>
#include <fstream>
#include <vector>
#include <type_traits>
#include <cassert>
#include <iostream>
#include <argp.h>

namespace ML {

    // --- Data Helper Functions ---
    // Floating Point Compare Epsilon
    constexpr float EPSILON = 0.001;


    // --- Argument Parsing ---
    // GCC argument input struct
    class arguments {
        public:
            bool debug;
            bool verify;
            bool singleLayer;
            int layerNum;        
            std::filesystem::path basePath;
            char *args[1];
    };

    // Args handling class (singlton)
    class Args {
        public:
            Args(Args &other) = delete;
            void operator=(const Args &) = delete;

            static Args& getInst() {
                static Args inst;
                return inst;
            }

            void parseArgs(int argc, char **argv);

        private:
            Args() : debug(false), verify(false), singleLayer(false), layerNum(-1), basePath("") {}

        public:
            // Options
            bool debug;
            bool verify;
            bool singleLayer;
            int layerNum;        
            std::filesystem::path basePath;
            std::string version;

        private:
            struct arguments _args;
    };

    // Metaprogramming type helpers
    template <typename T> struct remove_all_pointers{
        typedef T type;
    };

    template <typename T> struct remove_all_pointers<T*>{
        typedef typename remove_all_pointers<T>::type type;
    };


    // --- Implmentation ---

    // --- Data Helper Functions ---
    // Type cast helpers
    template<typename T>
    inline T castData(const void* data) { return (T) data; }

    // Recursive allocator, takes a raw array of dimentsions and the base type for allocation
    template<typename T_BASE>
    T_BASE* allocArray(const std::size_t* dims, const std::size_t dimsLen, const std::size_t dimIndex = 0) {
        static_assert(!std::is_pointer<T_BASE>(), "Cannot allocate pointer type values (arrays)");
        
        // Recursively allocate multidimentional arrays
        if (dimsLen > 1) {
            T_BASE** data = new T_BASE*[dims[dimIndex]];

            for (std::size_t i = 0; i < dims[dimIndex]; i++) {
                if (dimIndex < (dimsLen - 1)) {
                    data[i] = allocArray<T_BASE>(dims, dimsLen, dimIndex + 1);
                } else {
                    data[i] = 0;
                }
            }

            return reinterpret_cast<T_BASE*>(data);
        // Handle 1D arrays gracefully
        } else {
            return new T_BASE[dims[dimIndex]];
        }
    }

    // Take a vector of dims, takes the final array type as a template
    template<typename T>
    inline T allocArray(const std::vector<std::size_t>& dims, const std::size_t dimIndex = 0) {
        static_assert(std::is_pointer<T>(), "Cannot allocate non-pointer values (arrays)");
        // assert(std::rank<T>() == dims.size() && "Array type does not have the same rank as the dims provided");
        
        typedef typename remove_all_pointers<T>::type T_BASE;
        return reinterpret_cast<T>(allocArray<T_BASE>(dims.data(), dims.size(), dimIndex));
    }


    // --- Data array deallocation helpers ---
    template<typename T_BASE>
    void freeArray(T_BASE* data, const std::size_t* dims, const std::size_t dimsLen, const std::size_t dimIndex = 0) {
        static_assert(!std::is_pointer<T_BASE>(), "Cannot deallocate non-pointer values (arrays)");
        
        // Recursively free the multidimentional array as needed
        if (dimsLen > 1) {
            T_BASE** dataCast = reinterpret_cast<T_BASE**>(data);
            
            for (std::size_t i = 0; i < dims[dimIndex]; i++) {
                if (dimIndex < dimsLen - 2) {
                    freeArray<T_BASE>(dataCast[i], dims, dimsLen, dimIndex + 1);
                } else { 
                    delete [] dataCast[i];
                }
            }
        // Handle 1D Arrays
        } else {
            delete [] data;
        }
    }

    // Take a vector of dims
    template<typename T>
    inline void freeArray(T data, const std::vector<std::size_t>& dims, const std::size_t dimIndex = 0) {
        static_assert(std::is_pointer<T>(), "Cannot deallocate non-pointer values (arrays)");
        // assert(std::rank<T>() == dims.size() && "Array type does not have the same rank as the dims provided");

        typedef typename remove_all_pointers<T>::type T_BASE;
        freeArray<T_BASE>(reinterpret_cast<T_BASE*>(data), dims.data(), dims.size(), dimIndex);
    }


    // --- Compare Functions ---
    // Primary base compare function. Calls itself recursively as it decends structure
    template<typename T_BASE>
    float compareArray(const T_BASE* data1, const T_BASE* data2, const std::size_t* dims, const std::size_t dimsLen, const std::size_t dimIndex = 0) {
        static_assert(!std::is_pointer<T_BASE>(), "Cannot compare pointer type values (arrays)");
        double curr_diff = 0.0;
        double max_diff = 0.0;
        const T_BASE** dataCast1 = reinterpret_cast<const T_BASE**>(const_cast<T_BASE*>(data1));
        const T_BASE** dataCast2 = reinterpret_cast<const T_BASE**>(const_cast<T_BASE*>(data2));

        // Recurse as needed into each array
        for (std::size_t i = 0; i < dims[dimIndex]; i++) {
            if (dimIndex < dimsLen - 1) {
                // Recurse to compare another dimention down
                curr_diff = compareArray<T_BASE>(dataCast1[i], dataCast2[i], dims, dimsLen, dimIndex + 1);
            } else {
                // Check the values and get their absolute difference
                curr_diff = abs(data1[i] - data2[i]);
            }

            // Update our max difference if it is larger
            if (curr_diff > max_diff) {
                max_diff = curr_diff;
            }
        }

        return max_diff;
    }


    // Compares two LayerData arrays of size N and returns the maximum difference
    template<typename T>
    inline float compareArray(const T data1, const T data2, const std::vector<std::size_t>& dims, const std::size_t dimIndex = 0) {
        // assert(std::rank<T>() == dims.size() && "Array type does not have the same rank as the dims provided");
        
        typedef typename remove_all_pointers<T>::type T_BASE;
        return compareArray<T_BASE>(reinterpret_cast<T_BASE*>(data1), reinterpret_cast<T_BASE*>(data2), dims.data(), dims.size(), dimIndex);
    }


    // Performs a compare operation and checks if the max difference is within the provided epsilon
    template<typename T, typename EP_T = float>
    inline bool compareArrayWithin(const T data1, const T data2, const std::size_t* dims, const std::size_t dimsLen, EP_T epsilon, const std::size_t dimIndex = 0) {
        static_assert(!std::is_pointer<EP_T>(), "Cannot compare with pointer type (arrays) epsilon values");
        // assert(std::rank<T>() == dimsLen && "Array type does not have the same rank as the dims provided");

        typedef typename remove_all_pointers<T>::type T_BASE;
        return epsilon > compareArray<T_BASE>(reinterpret_cast<T_BASE*>(data1), reinterpret_cast<T_BASE*>(data2), dims, dimsLen, dimIndex);
    }

    // Performs a compare operation and checks if the max difference is within the provided epsilon
    template<typename T, typename EP_T = float>
    inline bool compareArrayWithin(const T data1, const T data2, const std::vector<std::size_t>& dims, EP_T epsilon, const std::size_t dimIndex = 0) {
        static_assert(!std::is_pointer<EP_T>(), "Cannot compare with pointer type (arrays) epsilon values");
        // assert(std::rank<T>() == dims.size() && "Array type does not have the same rank as the dims provided");

        return compareArrayWithin<T, EP_T>(data1, data2, dims.data(), dims.size(), epsilon, dimIndex);
    }

    
    // --- File Data Loading ---
    // Recursive array loading function that can load an array of data from multiple dimentions from a binary file
    template<typename T_BASE>
    void loadArrayData(std::ifstream& file, T_BASE* values, const std::size_t* dims, const std::size_t dimsLen, const std::size_t dimIndex = 0) {
        static_assert(!std::is_pointer<T_BASE>(), "Cannot load pointer type values (arrays)");

        // Read the values and recurse if needed
        if (dimsLen > 1) {
            T_BASE** valuesCast = reinterpret_cast<T_BASE**>(values);

            for (std::size_t i = 0; i < dims[dimIndex]; i++) {
                if (dimIndex < dimsLen - 2) {
                    // We do not care about the data pointer returned here since we have that already stored in a array
                    loadArrayData<T_BASE>(file, valuesCast[i], dims, dimsLen, dimIndex + 1);
                } else if (!file.read(reinterpret_cast<char*>(valuesCast[i]), sizeof(T_BASE) * dims[dimIndex + 1])) { // Read our values
                    std::cerr << "Failed to read data values from file" << std::endl;
                    assert(false && "Failed to read file data");
                }
            }
        // Handle 1D Arrays
        } else {
            if (!file.read(reinterpret_cast<char*>(values), sizeof(T_BASE) * dims[dimIndex])) { // Read our values
                std::cerr << "Failed to read data values from file" << std::endl;
                assert(false && "Failed to read file data");
            }
        }
    }


    // Entry point to loading data from a binary file into an array
    template<typename T>
    T loadArray(const std::filesystem::path& filepath, const std::vector<std::size_t>& dims) {
        static_assert(std::is_pointer<T>(), "Cannot load non-pointer values (arrays)");
        // assert(std::rank<T>() == dims.size() && "Array type does not have the same rank as the dims provided");
        
        // Open our file and check for issues
        std::ifstream file(filepath, std::ios::binary); // Create and open our file
        if (file.is_open()) {
            std::cout << "Opening binary file " << filepath << std::endl;
        } else {
            std::cerr << "Failed to open binary file " << filepath << std::endl;
        }

        // Allocate our array
        T values = allocArray<T>(dims);

        // Load the data
        typedef typename remove_all_pointers<T>::type T_BASE;
        loadArrayData<T_BASE>(file, reinterpret_cast<T_BASE*>(values), dims.data(), dims.size());
        return values;
    }

    // --- Copy Helpers ---
    // Deep copy an array by allocating a new one of the same size and recursively copying values
    template<typename T_BASE>
    void copyArray(const T_BASE* array, T_BASE* newArray, const std::size_t* dims, const std::size_t dimsLen, const std::size_t dimIndex = 0) {
        static_assert(!std::is_pointer<T_BASE>(), "Cannot copy pointer type values (arrays)");
        const T_BASE** arrayCast = reinterpret_cast<const T_BASE**>(const_cast<T_BASE*>(array));
        T_BASE** newArrayCast = reinterpret_cast<T_BASE**>(newArray);

        // Copy each value
        for (std::size_t i = 0; i < dims[dimIndex]; i++) {
            if (dimIndex < (dimsLen - 1)) {
                copyArray<T_BASE>(arrayCast[i], newArrayCast[i], dims, dimsLen, dimIndex + 1);
            } else {
                newArray[i] = array[i];
            }
        }
    }


    // Create a copy of an array, allocate a new array and recurse all values
    // Entry point
    template<typename T>
    inline void copyArray(const T array, T newArray, const std::vector<std::size_t>& dims) {
        static_assert(std::is_pointer<T>(), "Cannot copy non-pointer type values (arrays)");
        // assert(std::rank<T>() == dims.size() && "Array type does not have the same rank as the dims provided");

        typedef typename remove_all_pointers<T>::type T_BASE;
        copyArray<T_BASE>(reinterpret_cast<T_BASE*>(array), reinterpret_cast<T_BASE*>(newArray), dims.data(), dims.size(), 0);
    }


} // namespace ML