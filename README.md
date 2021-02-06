Pytorch Implementation of "Deep Region and Multi-label Learning for Facial Action Unit Detection"
We used the model architecture implemented in the paper. Many of the network codes are borrowed from 
https://github.com/AlexHex7/DRML_pytorch

The training loss functions are different from the original implementation because we are training on
BP4D+ dataset. Loss functions are implemented as "JAA-Net: Joint Facial Action Unit Detection and Face Alignment via ˆ
Adaptive Attention" (https://github.com/ZhiwenShao/PyTorch-JAANet).
