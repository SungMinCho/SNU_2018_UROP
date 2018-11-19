/**
 * Copyright 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NNAPI_MODEL_BUILDER_H
#define NNAPI_MODEL_BUILDER_H

#include <android/NeuralNetworks.h>
#include <vector>
#include <map>

#define LOG_TAG "MNIST_NNAPI"

using std::vector;
using std::map;

class Operand {
public:
    Operand() {};
    explicit Operand(vector<uint32_t*> &global_dim_arrays, vector<uint32_t> dim, uint32_t index, OperandCode operandCode);
    explicit Operand(vector<uint32_t*> &global_dim_arrays, vector<uint32_t> dim, uint32_t index, OperandCode operandCode, size_t offset, size_t length);
    ~Operand() {};

    void setConstant(void* buffer, size_t length);

    vector<uint32_t> dim;
    uint32_t* dim_array;
    ANeuralNetworksOperandType type;
    uint32_t index;
    OperandCode operandCode;

    bool isConstant;
    // offset and length in the file if this is not constant and the value is in a file
    size_t offset;
    size_t length;
    // buffer and length if it is constant
    void* constant_value_buffer;
    size_t constant_value_buffer_length;

    size_t byteLength();
    size_t dataLength();
};

class Operation {
public:
    Operation() {};
    explicit Operation(ANeuralNetworksOperationType operationType, vector<Operand> inputs, Operand output);
    ~Operation() {};

    ANeuralNetworksOperationType operationType;
    vector<Operand> inputs;
    Operand output;
};


// Right now suppose there is only one input (like image classification)
class ModelBuilder {
public:
    explicit ModelBuilder(size_t size, int protect, int fd, size_t offset);
    ~ModelBuilder();

    void addOperand(int index, vector<uint32_t>& dim, OperandCode operandCode);
    void addOperand(int index, vector<uint32_t>& dim, OperandCode operandCode, size_t offset, size_t length);
    void addConstantOperand(int index, vector<uint32_t>& dim, OperandCode operandCode, void* buffer, size_t length);
    Operand& getOperand(int index);
    void addFCLayer(int inputindex, int windex, int bindex, int activationindex, int outputindex);
    void addSoftmaxLayer(int inputindex, int betaindex, int outputindex);
    void parse(size_t size, int protect, int fd, size_t offset);
    bool CreateCompiledModel();
    // Rigiht now we suppose input = float[] , output = float[]
    bool Compute(float* input, float* output);

private:
    map<int, Operand> operands;
    vector<Operation> operations;
    Operand input_operand;
    Operand output_operand;
    uint32_t input_index;
    uint32_t output_index;

    vector<uint32_t*> global_dim_arrays;
    vector<void*> constants;
    vector<void*> constant_arrays;
    bool constructSuccess;

    ANeuralNetworksModel *model_;
    ANeuralNetworksCompilation *compilation_;
    ANeuralNetworksMemory *memoryModel_;
    ANeuralNetworksMemory *memoryInput_;
    ANeuralNetworksMemory *memoryOutput_;

    int inputTensorFd_;
    int outputTensorFd_;
};


#endif  // NNAPI_MODEL_BUILDER_H
