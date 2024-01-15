# SCSC
This repo is the official implementation of "SCSC: Spatial Cross-scale Convolution Module to Strengthen both CNNs and Transformers"

This paper presents a module, Spatial Cross-scale Convolution (SCSC), which is verified to be effective in improving both CNNs and Transformers. Nowadays, CNNs and Transformers have been successful in a variety of tasks. Especially for Transformers, increasing works achieve state-of-the-art performance in the computer vision community. Therefore, researchers start to explore the mechanism of those architectures. Large receptive fields, sparse connections, weight sharing, and dynamic weight have been considered keys to designing effective base models. However, there are still some issues to be addressed: large dense kernels and self-attention are inefficient, and large receptive fields make it hard to capture local features. Inspired by the above analyses and to solve the mentioned problems, in this paper, we design a general module taking in these design keys to enhance both CNNs and Transformers. SCSC introduces an efficient spatial cross-scale encoder and spatial embed module to capture assorted features in one layer. On the face recognition task, FaceResNet with SCSC can improve 2.7% with 68% fewer FLOPs and 79% fewer parameters. On the ImageNet classification task, Swin Transformer with SCSC can achieve even better performance with 22% fewer FLOPs, and ResNet with CSCS can improve 5.3% with similar complexity. Furthermore, a traditional network (e.g., ResNet) embedded with SCSC can match Swin Transformer's performance.


## Install
refer to [Swin](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md)

## Train
sh run.sh

## License
SCSC is released under the Apache 2.0 license.

