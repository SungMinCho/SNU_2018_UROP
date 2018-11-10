# SNU_2018_UROP
2018-2 UROP at SNU

## References
**Papers**
* [Distilling the Knowledge in a Neural Network](https://www.cs.toronto.edu/~hinton/absps/distillation.pdf)
* [SlimNets: An Exploration of Deep Model Compression and Acceleration](https://arxiv.org/pdf/1808.00496v1.pdf)
* [A Survey of Model Compression and Acceleration for Deep Neural Networks](https://arxiv.org/pdf/1710.09282.pdf)
* [DeepX : A Software Accelerator for Low-Power Deep Learning Inference on Mobile Devices](https://ix.cs.uoregon.edu/~jiao/papers/ipsn16.pdf)
* [DeepMon: Mobile GPU-based Deep Learning Framework for Continuous Vision Applications](https://nsr.cse.buffalo.edu/mobisys_2017/papers/pdfs/mobisys17-paper07.pdf)
* [Deep Learning for the Internet of Things](https://cse.buffalo.edu/~lusu/papers/Computer2018.pdf)
* [A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf)
* [FastDeepIoT: Towards Understanding and Optimizing Neural Network Execution Time on Mobile and Embedded Devices](https://arxiv.org/pdf/1809.06970.pdf)
* [Towards Evolutional Compression](https://arxiv.org/pdf/1707.08005.pdf)
* [Model Compression and Acceleration for Deep Neural Networks](http://cwww.ee.nctu.edu.tw/~cfung/docs/learning/cheng2018DNN_model_compression_accel.pdf)
---
* [AI Benchmark: Running Deep Neural Networks on Android Smartphones](https://arxiv.org/pdf/1810.01109v2.pdf)
```
Discussion Summary
현재 안드로이드 위에서 딥 러닝 돌리는 가장 편한 방법은 tensorflow mobile 이다. 나온지 2년이 넘었고 이슈들이 많이 해결이 됐다.
tensorflow lite 는 아직 추천하지 않는다. MobileNet 이나 Inception 으로 image classification 하는 간단한 task 외에는 문제가 있을 수 있다.
Tf mobile 에서 Tf lite 로 옮기는 건 쉽다. 안드로이드 인터페이스가 비슷하고, 모델 파일만 pb 포맷을 tflite 포맷으로 바꾸면 되니까.
그러니 나중에 tf lite 서포트가 더 좋아지면 옮기면 되겠다.
Caffe2 를 비롯한 다른 덜 유명한 프레임워크들은 커뮤니티가 작기 때문에 튜토리얼이나 문제 해결이 별로 없고 문제를 맞닥뜨리면 깃헙에 이슈를 올려서 해결해야 할 수도 있다

현재는 Kirin 970 칩셋 달고 있는 huawei 장비가 제일 빠르다. 하지만 아직 다른 칩셋들 나와봐야 안다. (The real situation will become clear at the beginning of the next year when the first devices with the Kirin 980, the MediaTek P80 and the next Qualcomm and Samsung Exynos premium SoCs will appear on the market)

quantized network 의 두 가지 미래를 생각해 볼 수 있다.
1. quantization 문제가 대부분 해결되고 대부분의 네트워크가 quantize 된다
2. float network 를 지원하는 NPU 의 성능이 점점 좋아져서 quantization 할 필요가 없을 수도 있다.
둘 중 미래가 어느 쪽으로 전개 될지 알기 어렵다.

메이저한 소프트웨어와 하드웨어, 프레임워크와 칩셋이 새로 나올때마다 벤치마크가 바뀔 것이다.
http://ai-benchmark.com
여기에 월마다 최신 벤치마크 결과를 업데이트할 예정.
```
* and possibly more... (will add later)

## Framework

* [TensorFlow](https://www.tensorflow.org)
* [Keras](https://keras.io/)
* [TensorFlow Lite](https://www.tensorflow.org/lite)
* [NNAPI](https://developer.android.com/ndk/guides/neuralnetworks/)
* possibly more
