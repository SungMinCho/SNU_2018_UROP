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

#define LOG_TAG "MNIST_NNAPI"

using std::vector;

class Dim {
public:
    explicit Dim(vector<uint32_t> dims);
    ~Dim();

    vector<uint32_t> dims;
};

class Operand {
public:
    explicit Operand(Dim dim, ANeuralNetworksOperandType type, uint32_t index);
    explicit Operand(Dim dim, ANeuralNetworksOperandType type, uint32_t index, size_t offset, size_t length);
    ~Operand();

    Dim dim;
    ANeuralNetworksOperandType type;
    uint32_t index;
    // offset >= 0 means it's a constant whose value is in memory (offset ~ offset + length)
    size_t offset;
    size_t length;

    size_t byteLength();
    size_t dataLength();
};

class Operation {
public:
    explicit Operation(ANeuralNetworksOperationType operationType, vector<Operand> inputs, Operand output);
    ~Operation();

    ANeuralNetworksOperationType operationType;
    vector<Operand> inputs;
    Operand output;
};


class ModelParser {
public:
    explicit ModelParser(size_t size, int protect, int fd, size_t offset);

    vector<Operand> operands;
    vector<Operation> operations;
    Operand input_operand;
    Operand output_operand;
};

// Right now suppose there is only one input (like image classification)
class ModelBuilder {
public:
    explicit ModelBuilder(size_t size, int protect, int fd, size_t offset);
    ~ModelBuilder();

    bool CreateCompiledModel();
    // Rigiht now we suppose input = float[] , output = float[]
    bool Compute(float* input, float* output);

private:
    vector<Operand> operands;
    vector<Operation> operations;
    Operand input_operand;
    Operand output_operand;
    uint32_t input_index;
    uint32_t output_index;

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