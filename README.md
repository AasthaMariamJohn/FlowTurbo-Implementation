FlowTurbo-Implementation

Implementation of the paper "FlowTurbo: Towards Real-time Flow-Based Image Generation with Velocity Refiner"
By Wenliang Zhao, Minglei Shi, Xumin Yu, Jie Zhou, Jiwen Lu
For the IPR course project, I have attempted to implement the codes given in their paper and observed the results.

FlowTurbo is designed to accelerate flow-based generative models using a new velocity refiner technique. By optimizing the sampling process, the paper claims that FlowTurbo achieves both significant acceleration and high-quality image generation.

This implementation achieved an acceleration ratio of 51.67% to 55% , and a FID score of 3.54 on class-conditional image generation tasks. These results are close to the paper's reported acceleration ratio of 53.1% to 58.3% and a FID of 2.11 on ImageNet (256x256) with 100 ms/img  FID of 3.93 with 38 (ms / img).

Below are some example images generated using FlowTurbo from our implementation:

![image](https://github.com/user-attachments/assets/c236b4c0-70a2-44fb-aa12-9a591001dc4d)


![image](https://github.com/user-attachments/assets/f9f3f3f3-a4e1-4d4d-bc06-593a7b0dcd94)

![image](https://github.com/user-attachments/assets/a9701128-87cd-4f53-8c4f-e6ce0aa740b3)

For a step-by-step walkthrough to run this implementation, refer to the link to instructions below.

https://colab.research.google.com/drive/1d3eIHNEI_TuifnnCgueJpgNL2uoiHmRO?usp=sharing

This guide will walk you through:

    Setting up - Cloning the required datasets and codes
    Installation - Dependencies and environment setup.
    Training - How to train and sample the refiner.
    Inference and evaluation - Running the model to generate new images.

Acknowledgments

This project is built upon the work presented in the FlowTurbo paper and is used for educational purposes for the IPR course.
@article{zhao2024flowturbo,
  title={FlowTurbo: Towards Real-time Flow-Based Image Generation with Velocity Refiner},
  author={Zhao, Wenliang and Shi, Minglei and Yu, Xumin and Zhou, Jie and Lu, Jiwen},
  journal={NeurIPS},
  year={2024}
}
