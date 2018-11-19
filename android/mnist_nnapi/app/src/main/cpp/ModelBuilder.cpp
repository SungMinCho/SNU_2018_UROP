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

void ModelBuilder::addOperand(int index, vector<uint32_t>& dim, OperandCode operandCode) {
    operands[index] = Operand(global_dim_arrays, dim, index, operandCode);
}

void ModelBuilder::addOperand(int index, vector<uint32_t>& dim, OperandCode operandCode,
                              size_t offset, size_t length) {
    operands[index] = Operand(global_dim_arrays, dim, index, operandCode, offset, length);
}

void ModelBuilder::addConstantOperand(int index, vector<uint32_t>& dim, OperandCode operandCode,
                                      void* buffer, size_t length) {
    addOperand(index, dim, operandCode);
    operands[index].setConstant(buffer, length); 
}

Operand& ModelBuilder::getOperand(int index) { return operands[index]; }

void ModelBuilder::addFCLayer(int inputindex, int windex, int bindex, 
                              int activationindex, int outputindex) {
    Operand& input_operand = getOperand(inputindex);
    Operand& w_operand = getOperand(windex);
    Operand& b_operand = getOperand(bindex);
    Operand& activation_operand = getOperand(activationindex);
    Operand& output_operand = getOperand(outputindex);

    vector<Operand> fc_inputs;
    fc_inputs.push_back(input_operand);
    fc_inputs.push_back(w_operand);
    fc_inputs.push_back(b_operand);
    fc_inputs.push_back(activation_operand);

    Operation fc = Operation(ANEURALNETWORKS_FULLY_CONNECTED,
            fc_inputs, output_operand);

    operations.push_back(fc);
}

void ModelBuilder::addSoftmaxLayer(int inputindex, int betaindex, int outputindex) {
  Operand& input_operand = getOperand(inputindex);
  Operand& beta_operand = getOperand(betaindex);
  Operand& output_operand = getOperand(outputindex);

  vector<Operand> inputs;
  inputs.push_back(input_operand);
  inputs.push_back(beta_operand);

  Operation softmax = Operation(ANEURALNETWORKS_SOFTMAX,
      inputs, output_operand);

  operations.push_back(softmax);
}

void ModelBuilder::parse(size_t size, int protect, int fd, size_t offset) {
    // ignore offset for now
    // suppose modelfile is an array of float value
    /*
    float* modelfile = reinterpret_cast<float *>(mmap(nullptr, size * sizeof(float),
        PROT_READ, MAP_SHARED, fd, 0));
    float* modelfilesave = modelfile;
    float* modelfileend = modelfile + size;

    for(; modelfile < modelfileend; modelfile++) {
      float f = *modelfile;
    }
    munmap(modelfilesave, size * sizeof(float));
    */
    size_t fileread = 0;

    vector<uint32_t> input_dim = {1, 784};
    addOperand(0, input_dim, ANEURALNETWORKS_TENSOR_FLOAT32);
    input_operand = getOperand(0);
    input_index = 0;

    vector<uint32_t> w1_dim = {100, 784};
    addOperand(1, w1_dim, ANEURALNETWORKS_TENSOR_FLOAT32, fileread, 100*784*4); 
    fileread += 100*784*4;

    vector<uint32_t> b1_dim = {100};
    addOperand(2, b1_dim, ANEURALNETWORKS_TENSOR_FLOAT32, fileread, 100*4); 
    fileread += 100*4;

    vector<uint32_t> activation_dim = {};
    FuseCode *fusecode = new FuseCode;
    *fusecode = ANEURALNETWORKS_FUSED_NONE;
    constants.push_back(fusecode);
    addConstantOperand(3, activation_dim, ANEURALNETWORKS_INT32, fusecode, sizeof(*fusecode)); 

    vector<uint32_t> out1_dim = {1, 100};
    addOperand(4, out1_dim, ANEURALNETWORKS_TENSOR_FLOAT32); 

    addFCLayer(0, 1, 2, 3, 4);
    
    vector<uint32_t> w2_dim = {100, 100};
    addOperand(5, w2_dim, ANEURALNETWORKS_TENSOR_FLOAT32, fileread, 100*100*4); 
    fileread += 100*100*4;

    vector<uint32_t> b2_dim = {100};
    addOperand(6, b2_dim, ANEURALNETWORKS_TENSOR_FLOAT32, fileread, 100*4); 
    fileread += 100*4;

    vector<uint32_t> out2_dim = {1, 100};
    addOperand(7, out2_dim, ANEURALNETWORKS_TENSOR_FLOAT32); 

    addFCLayer(4, 5, 6, 3, 7);

    vector<uint32_t> w3_dim = {10, 100};
    addOperand(8, w3_dim, ANEURALNETWORKS_TENSOR_FLOAT32, fileread, 10*100*4); 
    fileread += 10*100*4;

    vector<uint32_t> b3_dim = {10};
    addOperand(9, b3_dim, ANEURALNETWORKS_TENSOR_FLOAT32, fileread, 10*4); 
    fileread += 10*4;

    vector<uint32_t> out3_dim = {1, 10};
    addOperand(10, out3_dim, ANEURALNETWORKS_TENSOR_FLOAT32);
    output_operand = getOperand(10);

    addFCLayer(7, 8, 9, 3, 10);

    vector<uint32_t> beta_dim = {};
    float* beta_value = new float;
    *beta_value = 1.0;
    constants.push_back(beta_value);
    addConstantOperand(11, beta_dim, ANEURALNETWORKS_FLOAT32, beta_value, sizeof(*beta_value));

    vector<uint32_t> out4_dim = {1, 10};
    addOperand(12, out4_dim, ANEURALNETWORKS_TENSOR_FLOAT32);
    output_operand = getOperand(12);
    output_index = 12;

    addSoftmaxLayer(10, 11, 12);

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



ModelBuilder::ModelBuilder(size_t size, int protect, int fd, size_t offset)
{
    parse(size, protect, fd, offset);

    constructSuccess = false;
    int32_t status = ANEURALNETWORKS_NO_ERROR;
    status = ANeuralNetworksMemory_createFromFd(size+offset, protect, fd, 0, &memoryModel_);
    if(!checkStatus(status, "create memory from fd failed")) return;

    inputTensorFd_ = ASharedMemory_create("input", input_operand.byteLength());
    outputTensorFd_ = ASharedMemory_create("output", output_operand.byteLength());

    status = ANeuralNetworksMemory_createFromFd(input_operand.byteLength(),
            PROT_READ, inputTensorFd_, 0, &memoryInput_);
    if(!checkStatus(status, "input createfromfd failed")) return;

    status = ANeuralNetworksMemory_createFromFd(output_operand.byteLength(),
            PROT_READ, outputTensorFd_, 0, &memoryOutput_);
    if(!checkStatus(status, "output createfromfd failed")) return;

    /*
    std::ostringstream ss;
    ss << "memoryInput_ " << (void const*) memoryInput_
                          << " memoryOutput_ " << (void const*) memoryOutput_;
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, ss.str().c_str());
     */

    constructSuccess = true;
}


bool ModelBuilder::CreateCompiledModel() {
    if(!constructSuccess) {
      __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "not constructed");
      return false;
    }

    int32_t status;

    status = ANeuralNetworksModel_create(&model_);
    if(!checkStatus(status, "model create failed")) return false;

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

    // insert input
    float *inputTensorPtr = reinterpret_cast<float*>(mmap(nullptr, input_operand.byteLength(),
            PROT_READ | PROT_WRITE, MAP_SHARED, inputTensorFd_, 0));
    float *inputTensorPtrSave = inputTensorPtr;
    for(int i = 0; i < input_operand.dataLength(); i++) {
      *inputTensorPtr = input[i];
      inputTensorPtr++;
    }
    munmap(inputTensorPtrSave, input_operand.byteLength());

    // second argument 0 means the 0th (first) input from model Input list
    // right now we only have one input
    // but why is type nullptr...
    //status = ANeuralNetworksExecution_setInputFromMemory(execution, 0, nullptr,
    //        memoryInput_, 0, input_operand.byteLength());
    status = ANeuralNetworksExecution_setInput(execution, 0, nullptr,
            input, input_operand.byteLength());
    if(!checkStatus(status, "set input from memory failed")) return false;

    //status = ANeuralNetworksExecution_setOutputFromMemory(execution, 0, nullptr,
    //        memoryOutput_, 0, output_operand.byteLength());
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


    float *outputTensorPtr = reinterpret_cast<float *>(mmap(nullptr, output_operand.byteLength(),
            PROT_READ, MAP_SHARED, outputTensorFd_, 0));

    for(int i = 0; i < output_operand.dataLength(); i++) {
      //output[i] = outputTensorPtr[i];
    }

    munmap(outputTensorPtr, output_operand.byteLength());


    return true;
}

ModelBuilder::~ModelBuilder() {
    ANeuralNetworksCompilation_free(compilation_);
    ANeuralNetworksModel_free(model_);
    ANeuralNetworksMemory_free(memoryModel_);
    ANeuralNetworksMemory_free(memoryInput_);
    ANeuralNetworksMemory_free(memoryOutput_);
    close(inputTensorFd_);
    close(outputTensorFd_);

    for(auto &d : global_dim_arrays) delete[] d;
    for(auto &c : constants) delete c;
    for(auto &c : constant_arrays) delete[] c;
}
