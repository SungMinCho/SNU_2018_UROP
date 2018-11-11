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
#include <unistd.h>
#include <map>


Dim::Dim(vector<uint32_t> dims) : dims(dims) {}

Operand::Operand(Dim dim, ANeuralNetworksOperandType type, uint32_t index) : dim(dim), type(type), index(index), offset(-1) {}
Operand::Operand(Dim dim, ANeuralNetworksOperandType type, uint32_t index, size_t offset, size_t length) : dim(dim), type(type), index(index), offset(offset), length(length) {}
size_t Operand::byteLength() {
   // return byte length of this operand
   return 0; // implment later
}
size_t Operand::dataLength() {
   // return length of this operand (1 = each data)
   return 0;
}

Operation::Operation(ANeuralNetworksOperationType operationType, vector<Operand> inputs,
                     Operand output) : operationType(operationType),
                                       inputs(inputs),
                                       output(output) {}
ModelParser::ModelParser(size_t size, int protect, int fd, size_t offset) {
   // ignore offset for now
   // suppose modelfile is an array of float values

   float* modelfile = reinterpret_cast<float *>(mmap(nullptr, size * sizeof(float),
       PROT_READ, MAP_SHARED, fd, 0));
   float* modelfilesave = modelfile;
   float* modelfileend = modelfile + size;

   for(; modelfile < modelfileend; modelfile++) {
      float f = *modelfile;
   }


   munmap(modelfilesave, size * sizeof(float));
}

bool checkStatus(int32_t status, const char *msg) {
   if(status != ANEURALNETWORKS_NO_ERROR) {
      __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, msg);
      return false;
   }
   return true;
}

ModelBuilder::ModelBuilder(size_t size, int protect, int fd, size_t offset)
{
   ModelParser parser = ModelParser(size, protect, fd, offset);
   operands = parser.operands;
   operations = parser.operations;
   input_operand = parser.input_operand;
   output_operand = parser.output_operand;
   input_index = input_operand.index;
   output_index = output_operand.index;

   constructSuccess = false;
   int32_t status;
   //status = ANeuralNetworksMemory_createFromFd(size + offset, protect, fd, 0, &memoryModel_);
   status = ANeuralNetworksMemory_createFromFd(size, protect, fd, offset, &memoryModel_);
   if(!checkStatus(status, "create memory from fd failed")) return;

   inputTensorFd_ = ASharedMemory_create("input", input_operand.byteLength());
   outputTensorFd_ = ASharedMemory_create("output", output_operand.byteLength());

   status = ANeuralNetworksMemory_createFromFd(input_operand.byteLength(),
           PROT_READ, inputTensorFd_, 0, &memoryInput_);
   if(!checkStatus(status, "input createfromfd failed")) return;

   status = ANeuralNetworksMemory_createFromFd(output_operand.byteLength(),
           PROT_READ, outputTensorFd_, 0, &memoryOutput_);
   if(!checkStatus(status, "output createfromfd failed")) return;

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

   for(Operand o : operands) {
      indexDict[o.index] = opIndex++;
      status = ANeuralNetworksModel_addOperand(model_, &o.type);
      if(!checkStatus(status, "add operand failed")) return false;

      if(o.offset >= 0) {
         status = ANeuralNetworksModel_setOperandValueFromMemory(model_,
                 indexDict[o.index], memoryModel_, o.offset, o.length);
         if(!checkStatus(status, "set operand value from memory failed")) return false;
      }
   }

   // activation??
   // later

   for(Operation op : operations) {
      vector<uint32_t> inputs;
      for(Operand i : op.inputs) {
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
   status = ANeuralNetworksExecution_setInputFromMemory(execution, 0, nullptr,
           memoryInput_, 0, input_operand.byteLength());
   if(!checkStatus(status, "set input from memory failed")) return false;

   status = ANeuralNetworksExecution_setOutputFromMemory(execution, 0, nullptr,
           memoryOutput_, 0, output_operand.byteLength());
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
      output[i] = outputTensorPtr[i];
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
}
