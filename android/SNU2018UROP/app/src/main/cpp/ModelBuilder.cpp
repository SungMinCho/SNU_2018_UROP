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
#include "ModelBuilder.h"

#include <android/log.h>
#include <android/sharedmem.h>
#include <sys/mman.h>
#include <string>
#include <sstream>
#include <unistd.h>
#include <map>

void log_operand(Operand o) {
  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "index");
  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, std::to_string(o.index).c_str());

  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "dim.size()");
  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, std::to_string(o.dim.size()).c_str());

  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "dimensionCount");
  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, std::to_string(o.type.dimensionCount).c_str());


  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "dimensions");
  for(int i = 0; i < o.type.dimensionCount; i++)
      __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, std::to_string(o.type.dimensions[i]).c_str());
  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "operandCode");
  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, std::to_string(o.operandCode).c_str());
}

size_t floatByteLength(vector<uint32_t>& dim) {
    size_t res = 4;
    for(auto &d : dim) res *= d;
    return res;
}

Operand::Operand(vector<uint32_t*> &global_dim_arrays, vector<uint32_t> dim, uint32_t index, OperandCode operandCode)
    : Operand(global_dim_arrays, dim, index, operandCode, -1, 0) {}
Operand::Operand(vector<uint32_t*> &global_dim_arrays, vector<uint32_t> dim_, uint32_t index_, OperandCode operandCode_, size_t offset_, size_t length_)
  : dim(dim_), index(index_), operandCode(operandCode_), isConstant(false), offset(offset_), length(length_) {
    uint32_t dimCount = dim.size();
    if(dimCount == 0) dim_array = nullptr;
    else {
        dim_array = new uint32_t[dimCount];
        for(int i = 0; i < dimCount; i++) dim_array[i] = dim[i];
        global_dim_arrays.push_back(dim_array);
        /*
        std::ostringstream ss;
        ss << "index " << index << " " << (void const*) dim_array;
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, ss.str().c_str());
        */
    }

    type = {
      .type = operandCode_,
      .dimensionCount = dimCount,
      .dimensions = dim_array,
      .scale = 0.0f,
      .zeroPoint = 0
    };
}

void Operand::setConstant(void* buffer, size_t length) {
  isConstant = true;
  constant_value_buffer = buffer;
  constant_value_buffer_length = length;
}

size_t Operand::byteLength() {
    // return byte length of this operand
    size_t bytesize = 1;
    if(type.type == ANEURALNETWORKS_TENSOR_FLOAT32) {
        bytesize = 4;
    } // add more later
    return bytesize * dataLength();
}
size_t Operand::dataLength() {
    // return length of this operand (1 = each data)
    size_t res = 1;
    for(auto &d : dim) {
        res *= d;
    }
    return res;
}

Operation::Operation(ANeuralNetworksOperationType operationType, vector<Operand> inputs,
                     Operand output) : operationType(operationType),
                                        inputs(inputs),
                                        output(output) {}

void ModelBuilder::addOperand(Index index, vector<uint32_t>& dim, OperandCode operandCode) {
    operands[index] = Operand(global_dim_arrays, dim, index, operandCode);
}

void ModelBuilder::addOperand(Index index, vector<uint32_t>& dim, OperandCode operandCode,
                              size_t offset, size_t length) {
    operands[index] = Operand(global_dim_arrays, dim, index, operandCode, offset, length);
}

void ModelBuilder::addConstantOperand(Index index, vector<uint32_t>& dim, OperandCode operandCode,
                                      void* buffer, size_t length) {
    addOperand(index, dim, operandCode);
    operands[index].setConstant(buffer, length); 
}

Operand& ModelBuilder::getOperand(Index index) { return operands[index]; }

void ModelBuilder::addFCLayer(Index input, Index weight, Index bias, Index activation, Index output) {
    Operand& input_operand = getOperand(input);
    Operand& w_operand = getOperand(weight);
    Operand& b_operand = getOperand(bias);
    Operand& activation_operand = getOperand(activation);
    Operand& output_operand = getOperand(output);

    vector<Operand> fc_inputs;
    fc_inputs.push_back(input_operand);
    fc_inputs.push_back(w_operand);
    fc_inputs.push_back(b_operand);
    fc_inputs.push_back(activation_operand);

    Operation fc = Operation(ANEURALNETWORKS_FULLY_CONNECTED,
            fc_inputs, output_operand);

    operations.push_back(fc);
}

void ModelBuilder::addSoftmaxLayer(Index input, Index beta, Index output) {
  Operand& input_operand = getOperand(input);
  Operand& beta_operand = getOperand(beta);
  Operand& output_operand = getOperand(output);

  vector<Operand> inputs;
  inputs.push_back(input_operand);
  inputs.push_back(beta_operand);

  Operation softmax = Operation(ANEURALNETWORKS_SOFTMAX,
      inputs, output_operand);

  operations.push_back(softmax);
}

void ModelBuilder::addConv2DLayer(Index input, Index filter, Index bias, Index padding,
                    Index widthStride, Index heightStride, Index activation, Index output){
    Operand& input_operand = getOperand(input);
    Operand& filter_operand = getOperand(filter);
    Operand& bias_operand = getOperand(bias);
    Operand& padding_operand = getOperand(padding);
    Operand& widthStride_operand = getOperand(widthStride);
    Operand& heightStride_operand = getOperand(heightStride);
    Operand& activation_operand = getOperand(activation);
    Operand& output_operand = getOperand(output);

    vector<Operand> inputs;
    inputs.push_back(input_operand);
    inputs.push_back(filter_operand);
    inputs.push_back(bias_operand);
    inputs.push_back(padding_operand);
    inputs.push_back(widthStride_operand);
    inputs.push_back(heightStride_operand);
    inputs.push_back(activation_operand);

    Operation conv2d = Operation(ANEURALNETWORKS_CONV_2D, inputs, output_operand);

    operations.push_back(conv2d);
}

void ModelBuilder::addMaxPool2DLayer(Index input, Index padding, Index widthStride, Index heightStride,
                       Index filterWidth, Index filterHeight, Index activation, Index output) {
    Operand& input_operand = getOperand(input);
    Operand& padding_operand = getOperand(padding);
    Operand& widthStride_operand = getOperand(widthStride);
    Operand& heightStride_operand = getOperand(heightStride);
    Operand& filterWidth_operand = getOperand(filterWidth);
    Operand& filterHeight_operand = getOperand(filterHeight);
    Operand& activation_operand = getOperand(activation);
    Operand& output_operand = getOperand(output);

    vector<Operand> inputs;
    inputs.push_back(input_operand);
    inputs.push_back(padding_operand);
    inputs.push_back(widthStride_operand);
    inputs.push_back(heightStride_operand);
    inputs.push_back(filterWidth_operand);
    inputs.push_back(filterHeight_operand);
    inputs.push_back(activation_operand);

    Operation max_pool_2d = Operation(ANEURALNETWORKS_MAX_POOL_2D, inputs, output_operand);

    operations.push_back(max_pool_2d);
}

void ModelBuilder::smartAddFCLayer(uint32_t nodenum, Index ACTIVATION, size_t& fileread) {
    assert(nodenum > 0);
    Index INPUT = output_index;

    vector<uint32_t> in_dim = output_operand.dim;
    uint32_t batch = in_dim[0];
    uint32_t in_nodenum = 1;
    for(int i = 1; i < in_dim.size(); i++) in_nodenum *= in_dim[i];

    vector<uint32_t> dim = {nodenum, in_nodenum};
    Index WEIGHT = getNewIndex();
    addOperand(WEIGHT, dim, ANEURALNETWORKS_TENSOR_FLOAT32, fileread, floatByteLength(dim));
    fileread += floatByteLength(dim);

    dim = {nodenum};
    Index BIAS = getNewIndex();
    addOperand(BIAS, dim, ANEURALNETWORKS_TENSOR_FLOAT32, fileread, floatByteLength(dim));
    fileread += floatByteLength(dim);

    dim = {batch, nodenum};
    Index OUT = getNewIndex();
    addOperand(OUT, dim, ANEURALNETWORKS_TENSOR_FLOAT32);
    output_index = OUT;
    output_operand = getOperand(OUT);

    addFCLayer(INPUT, WEIGHT, BIAS, ACTIVATION, OUT);
}

void ModelBuilder::smartAddSoftmaxLayer(Index BETA) {
    Index INPUT = output_index;
    Operand& INPUT_OPERAND = output_operand;
    assert(INPUT_OPERAND.dim.size() == 2);

    Index OUT = getNewIndex();
    addOperand(OUT, INPUT_OPERAND.dim, INPUT_OPERAND.operandCode);
    output_index = OUT;
    output_operand = getOperand(OUT);

    addSoftmaxLayer(INPUT, BETA, OUT);
}

void ModelBuilder::smartAddConv2DLayer(vector<uint32_t> filterDim, Index PADDING, Index WSTRIDE, Index HSTRIDE, Index ACTIVATION, size_t& fileread) {
    // PADDING, WSTRIDE, HSTRIDE, ACTIVATION are indexes

    Index INPUT = output_index;

    // filter [D_out, H, W, D_in]
    Index FILTER = getNewIndex();
    addOperand(FILTER, filterDim, ANEURALNETWORKS_TENSOR_FLOAT32, fileread, floatByteLength(filterDim));
    fileread += floatByteLength(filterDim);

    // bias
    vector<uint32_t> biasDim = {filterDim[0]}; // D_out
    Index BIAS = getNewIndex();
    addOperand(BIAS, biasDim, ANEURALNETWORKS_TENSOR_FLOAT32, fileread, floatByteLength(biasDim));
    fileread += floatByteLength(biasDim);

    // [B,H,W,D_in]
    vector<uint32_t> originalDim = output_operand.dim;
    // [B,H,W,D_out]
    vector<uint32_t> outDim = {originalDim[0], originalDim[1], originalDim[2], filterDim[0]};
    Index OUT = getNewIndex();
    addOperand(OUT, outDim, ANEURALNETWORKS_TENSOR_FLOAT32);
    output_index = OUT;
    output_operand = getOperand(OUT);

    addConv2DLayer(INPUT, FILTER, BIAS, PADDING, WSTRIDE, HSTRIDE, ACTIVATION, OUT);
}

void ModelBuilder::smartAddMaxPool2DLayer(Index PADDING, Index WSTRIDE, Index HSTRIDE, Index FILTERW, Index FILTERH, Index ACTIVATION) {
    int INPUT = output_index;

    vector<uint32_t> originalDim = output_operand.dim;
    uint32_t B = originalDim[0];
    uint32_t H = originalDim[1];
    uint32_t W = originalDim[2];
    uint32_t D = originalDim[3];

    int wstride = *(int*)(getOperand(WSTRIDE).constant_value_buffer);
    int hstride = *(int*)(getOperand(HSTRIDE).constant_value_buffer);
    int wfilter = *(int*)(getOperand(FILTERW).constant_value_buffer);
    int hfilter = *(int*)(getOperand(FILTERH).constant_value_buffer);

    // TODO : possible errors in cornercases?
    uint32_t newH = (H-hfilter) / hstride + 1;
    uint32_t newW = (W-wfilter) / wstride + 1;

    vector<uint32_t> dim = {B, newH, newW, D};
    Index OUT = getNewIndex();
    addOperand(OUT, dim, ANEURALNETWORKS_TENSOR_FLOAT32);
    output_index = OUT;
    output_operand = getOperand(OUT);

    addMaxPool2DLayer(INPUT, PADDING, WSTRIDE, HSTRIDE, FILTERW, FILTERH, ACTIVATION, OUT);
}

bool checkStatus(int32_t status, const char *msg) {
    if(status != ANEURALNETWORKS_NO_ERROR) {
      __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, msg);
      std::string s = std::to_string(status);
      __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, s.c_str());
      return false;
    }
    return true;
}

ModelBuilder::Index ModelBuilder::getNewIndex() {
    return index++;
}

ModelBuilder::ModelBuilder(size_t size, int protect, int fd, size_t offset)
        : filefd(fd), filesize(size), fileoffset(offset), index(0)
{
    constructSuccess = false;
    int32_t status;
    status = ANeuralNetworksMemory_createFromFd(size+offset, protect, fd, 0, &memoryModel_);
    if(!checkStatus(status, "create memory from fd failed")) return;

    constructSuccess = true;
}

bool ModelBuilder::CreateVGG16Model() {
    /*
     * Model Architecture
     *
     * VGG(
          (features): Sequential(
            (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU(inplace)
            (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): ReLU(inplace)
            (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (6): ReLU(inplace)
            (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (8): ReLU(inplace)
            (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (11): ReLU(inplace)
            (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (13): ReLU(inplace)
            (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (15): ReLU(inplace)
            (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (18): ReLU(inplace)
            (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (20): ReLU(inplace)
            (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (22): ReLU(inplace)
            (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (25): ReLU(inplace)
            (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (27): ReLU(inplace)
            (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (29): ReLU(inplace)
            (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          )
          (classifier): Sequential(
            (0): Linear(in_features=25088, out_features=4096, bias=True)
            (1): ReLU(inplace)
            (2): Dropout(p=0.5)
            (3): Linear(in_features=4096, out_features=4096, bias=True)
            (4): ReLU(inplace)
            (5): Dropout(p=0.5)
            (6): Linear(in_features=4096, out_features=1000, bias=True)
          )
        )


        Parameter shapes:
        torch.Size([64, 3, 3, 3])
        torch.Size([64])
        torch.Size([64, 64, 3, 3])
        torch.Size([64])
        torch.Size([128, 64, 3, 3])
        torch.Size([128])
        torch.Size([128, 128, 3, 3])
        torch.Size([128])
        torch.Size([256, 128, 3, 3])
        torch.Size([256])
        torch.Size([256, 256, 3, 3])
        torch.Size([256])
        torch.Size([256, 256, 3, 3])
        torch.Size([256])
        torch.Size([512, 256, 3, 3])
        torch.Size([512])
        torch.Size([512, 512, 3, 3])
        torch.Size([512])
        torch.Size([512, 512, 3, 3])
        torch.Size([512])
        torch.Size([512, 512, 3, 3])
        torch.Size([512])
        torch.Size([512, 512, 3, 3])
        torch.Size([512])
        torch.Size([512, 512, 3, 3])
        torch.Size([512])
        torch.Size([4096, 25088])
        torch.Size([4096])
        torch.Size([4096, 4096])
        torch.Size([4096])
        torch.Size([1000, 4096])
        torch.Size([1000])
     */

    size_t fileread = fileoffset;
    int index = 0;

    // constants

    vector<uint32_t> dim;
    FuseCode* fuseCode;

    dim = {};
    fuseCode = new FuseCode;
    *fuseCode = ANEURALNETWORKS_FUSED_NONE;
    constants.push_back(fuseCode);
    int ACTIVATION_NONE = getNewIndex();
    addConstantOperand(ACTIVATION_NONE, dim, ANEURALNETWORKS_INT32, fuseCode, sizeof(*fuseCode));

    dim = {};
    fuseCode = new FuseCode;
    *fuseCode = ANEURALNETWORKS_FUSED_RELU;
    constants.push_back(fuseCode);
    int RELU = getNewIndex();
    addConstantOperand(RELU, dim, ANEURALNETWORKS_INT32, fuseCode, sizeof(*fuseCode));

    dim = {};
    PaddingCode* paddingCode = new PaddingCode;
    *paddingCode = ANEURALNETWORKS_PADDING_SAME;
    constants.push_back(paddingCode);
    int PADDING_SAME = getNewIndex();
    addConstantOperand(PADDING_SAME, dim, ANEURALNETWORKS_INT32, paddingCode, sizeof(*paddingCode));

    dim = {};
    paddingCode = new PaddingCode;
    *paddingCode = ANEURALNETWORKS_PADDING_VALID;
    constants.push_back(paddingCode);
    int PADDING_VALID = getNewIndex();
    addConstantOperand(PADDING_VALID, dim, ANEURALNETWORKS_INT32, paddingCode, sizeof(*paddingCode));

    dim = {};
    int* intp = new int;
    *intp = 1;
    constants.push_back(intp);
    int ONE = getNewIndex();
    addConstantOperand(ONE, dim, ANEURALNETWORKS_INT32, intp, sizeof(*intp));

    dim = {};
    intp = new int;
    *intp = 2;
    constants.push_back(intp);
    int TWO = getNewIndex();
    addConstantOperand(TWO, dim, ANEURALNETWORKS_INT32, intp, sizeof(*intp));

    dim = {};
    float* floatp = new float;
    *floatp = 1.0;
    constants.push_back(floatp);
    int ONE_FLOAT = getNewIndex();
    addConstantOperand(ONE_FLOAT, dim, ANEURALNETWORKS_FLOAT32, floatp, sizeof(*floatp));

    // Input

    dim = {1, 224, 224, 3}; // [B,H,W,D]
    int INPUT = getNewIndex();
    addOperand(INPUT, dim, ANEURALNETWORKS_TENSOR_FLOAT32);
    // necessary for model construction
    input_operand = getOperand(INPUT);
    input_index = INPUT;
    // will change as we smart add layers
    output_operand = getOperand(INPUT);
    output_index = INPUT;

    // Features

    // filter dim : [D_out, H, W, D_in]
    vector<uint32_t> filterDim = {64, 3, 3, 3};
    smartAddConv2DLayer(filterDim, PADDING_SAME, ONE, ONE, RELU, fileread);

    filterDim = {64, 3, 3, 64};
    smartAddConv2DLayer(filterDim, PADDING_SAME, ONE, ONE, RELU, fileread);

    smartAddMaxPool2DLayer(PADDING_VALID, TWO, TWO, TWO, TWO, ACTIVATION_NONE);

    filterDim = {128, 3, 3, 64};
    smartAddConv2DLayer(filterDim, PADDING_SAME, ONE, ONE, RELU, fileread);

    filterDim = {128, 3, 3, 128};
    smartAddConv2DLayer(filterDim, PADDING_SAME, ONE, ONE, RELU, fileread);

    smartAddMaxPool2DLayer(PADDING_VALID, TWO, TWO, TWO, TWO, ACTIVATION_NONE);

    filterDim = {256, 3, 3, 128};
    smartAddConv2DLayer(filterDim, PADDING_SAME, ONE, ONE, RELU, fileread);

    filterDim = {256, 3, 3, 256};
    smartAddConv2DLayer(filterDim, PADDING_SAME, ONE, ONE, RELU, fileread);

    filterDim = {256, 3, 3, 256};
    smartAddConv2DLayer(filterDim, PADDING_SAME, ONE, ONE, RELU, fileread);

    smartAddMaxPool2DLayer(PADDING_VALID, TWO, TWO, TWO, TWO, ACTIVATION_NONE);

    filterDim = {512, 3, 3, 256};
    smartAddConv2DLayer(filterDim, PADDING_SAME, ONE, ONE, RELU, fileread);

    filterDim = {512, 3, 3, 512};
    smartAddConv2DLayer(filterDim, PADDING_SAME, ONE, ONE, RELU, fileread);

    filterDim = {512, 3, 3, 512};
    smartAddConv2DLayer(filterDim, PADDING_SAME, ONE, ONE, RELU, fileread);

    smartAddMaxPool2DLayer(PADDING_VALID, TWO, TWO, TWO, TWO, ACTIVATION_NONE);

    filterDim = {512, 3, 3, 512};
    smartAddConv2DLayer(filterDim, PADDING_SAME, ONE, ONE, RELU, fileread);

    filterDim = {512, 3, 3, 512};
    smartAddConv2DLayer(filterDim, PADDING_SAME, ONE, ONE, RELU, fileread);

    filterDim = {512, 3, 3, 512};
    smartAddConv2DLayer(filterDim, PADDING_SAME, ONE, ONE, RELU, fileread);

    smartAddMaxPool2DLayer(PADDING_VALID, TWO, TWO, TWO, TWO, ACTIVATION_NONE);

    // Classifier

    smartAddFCLayer(4096, RELU, fileread);
    smartAddFCLayer(4096, RELU, fileread);
    smartAddFCLayer(1000, ACTIVATION_NONE, fileread);

    // Softmax

    smartAddSoftmaxLayer(ONE_FLOAT);

    if(fileread - fileoffset != filesize) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "file read incorrect");
        return false;
    }

    return true;
}

bool ModelBuilder::CreateCompiledModel() {
    if(!constructSuccess) {
      __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "not constructed");
      return false;
    }

    int32_t status;

    status = ANeuralNetworksModel_create(&model_);
    if(!checkStatus(status, "model create failed")) return false;

    // model_ is assured to be created
    CreateVGG16Model();

    // index -> nnapi index
    std::map<uint32_t, uint32_t> indexDict;
    uint32_t opIndex = 0;

    for(auto &i : operands) {
      Operand o = i.second;
      indexDict[o.index] = opIndex++;

      status = ANeuralNetworksModel_addOperand(model_, &o.type);
      if(!checkStatus(status, "add operand failed")) return false;

      if(o.offset != -1) {
         status = ANeuralNetworksModel_setOperandValueFromMemory(model_,
                 indexDict[o.index], memoryModel_, o.offset, o.length);
         if(!checkStatus(status, "set operand value from memory failed")) return false;
      }
      else {
          if(o.isConstant == true) {
            status = ANeuralNetworksModel_setOperandValue(model_, indexDict[o.index],
                o.constant_value_buffer, o.constant_value_buffer_length);
            if(!checkStatus(status, "set operand value failed")) return false;
          }
          if(o.index == 3) {
            // naively take care of activation
          }
      }
    }

    // activation??
    // later

    for(Operation op : operations) {
      vector<uint32_t> inputs;
      for(Operand i : op.inputs) {
          //__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "input operand log");
          //log_operand(i);
          //__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "/////////////////////////////////");
          inputs.push_back(indexDict[i.index]);
      }
      uint32_t output = indexDict[op.output.index];
      status = ANeuralNetworksModel_addOperation(model_, op.operationType,
              inputs.size(), inputs.data(), 1, &output);
      if(!checkStatus(status, "add operation failed")) return false;
    }

    // Suppose 1 input 1 output
    uint32_t input = indexDict[input_index];
    uint32_t output = indexDict[output_index];
    status = ANeuralNetworksModel_identifyInputsAndOutputs(model_,
            1, &input, 1, &output);
    if(!checkStatus(status, "identify inputs and outputs failed")) return false;

    status = ANeuralNetworksModel_finish(model_);
    if(!checkStatus(status, "model finish failed")) return false;

    status = ANeuralNetworksCompilation_create(model_, &compilation_);
    if(!checkStatus(status, "compilation create failed")) return false;

    // compare different preferences later
    status = ANeuralNetworksCompilation_setPreference(compilation_,
            ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
    if(!checkStatus(status, "compilation set preference failed")) return false;

    status = ANeuralNetworksCompilation_finish(compilation_);
    if(!checkStatus(status, "compilation finish failed")) return false;

    return true;
}

bool ModelBuilder::Compute(float *input, float *output) {
    if(!constructSuccess) {
      __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "not constructed");
      return false;
    }

    int32_t status;

    ANeuralNetworksExecution *execution;
    status = ANeuralNetworksExecution_create(compilation_, &execution);
    if(!checkStatus(status, "execution create failed")) return false;

    status = ANeuralNetworksExecution_setInput(execution, 0, nullptr,
            input, input_operand.byteLength());
    if(!checkStatus(status, "set input from memory failed")) return false;

    status = ANeuralNetworksExecution_setOutput(execution, 0, nullptr,
            output, output_operand.byteLength());
    if(!checkStatus(status, "set output from memory failed")) return false;

    ANeuralNetworksEvent *event = nullptr;
    status = ANeuralNetworksExecution_startCompute(execution, &event);
    if(!checkStatus(status, "start compute failed")) return false;

    status = ANeuralNetworksEvent_wait(event);
    if(!checkStatus(status, "event wait failed")) return false;

    ANeuralNetworksEvent_free(event);
    ANeuralNetworksExecution_free(execution);

    return true;
}

ModelBuilder::~ModelBuilder() {
    ANeuralNetworksCompilation_free(compilation_);
    ANeuralNetworksModel_free(model_);
    ANeuralNetworksMemory_free(memoryModel_);

    for(auto &d : global_dim_arrays) delete[] d;
    for(auto &c : constants) delete c;
    for(auto &c : constant_arrays) delete[] c;
}
