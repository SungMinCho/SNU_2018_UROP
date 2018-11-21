#include <jni.h>
#include <string>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <android/sharedmem.h>
#include <sys/mman.h>
#include "ModelBuilder.h"


extern "C" JNIEXPORT jstring JNICALL
Java_sungmincho_snu2018urop_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

extern "C"
JNIEXPORT jlong
JNICALL
Java_sungmincho_snu2018urop_NNAPI_1Classifier_initModel(
        JNIEnv *env,
        jobject /* this */,
        jobject _assetManager,
        jstring _assetName) {
    // Get the file descriptor of the the model data file.
    AAssetManager *assetManager = AAssetManager_fromJava(env, _assetManager);
    const char *assetName = env->GetStringUTFChars(_assetName, NULL);
    AAsset *asset = AAssetManager_open(assetManager, assetName, AASSET_MODE_BUFFER);
    if(asset == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Failed to open the asset.");
        return 0;
    }
    env->ReleaseStringUTFChars(_assetName, assetName);
    off_t offset, length;
    int fd = AAsset_openFileDescriptor(asset, &offset, &length);
    AAsset_close(asset);
    if (fd < 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "Failed to open the model_data file descriptor.");
        return 0;
    }
    ModelBuilder* nn_model = new ModelBuilder(length, PROT_READ, fd, offset);
    if (!nn_model->CreateCompiledModel()) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "Failed to prepare the model.");
        return 0;
    }

    return (jlong)(uintptr_t)nn_model;
}

extern "C"
JNIEXPORT void
JNICALL
Java_sungmincho_snu2018urop_NNAPI_1Classifier_startCompute(
        JNIEnv *env,
        jobject /* this */,
        jlong _nnModel,
        jobject inputByteBuffer,
        jfloatArray outputArray) {
    ModelBuilder* nn_model = (ModelBuilder*) _nnModel;

    float* input = (float*) env->GetDirectBufferAddress(inputByteBuffer);

    float result_array[1000];
    nn_model->Compute(input, result_array);
    env->SetFloatArrayRegion(outputArray, 0, 1000, result_array);
}

extern "C"
JNIEXPORT void
JNICALL
Java_sungmincho_snu2018urop_NNAPI_1Classifier_destroyModel(
        JNIEnv *env,
        jobject /* this */,
        jlong _nnModel) {
    ModelBuilder* nn_model = (ModelBuilder*) _nnModel;
    delete(nn_model);
}
