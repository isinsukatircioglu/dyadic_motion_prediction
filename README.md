# Dyadic Human Motion Prediction

This is the code for the paper “Dyadic Human Motion Prediction”. Our code is built on top of the code from “History Repeats Itself: Human Motion Prediction via Motion Attention” (https://github.com/wei-mao-2019/HisRepItself). Inside ‘data’ folder, we provide the 3D poses of the test subjects. We will publish the full dataset (training/validation/test) with the videos and the corresponding 3D ground truth body poses later on.

## Dependencies
Tested on NVIDIA Tesla V100-SXM2-32GB with
- Pytorch 1.7.1+cu110
- Python 3.6.9

## Training
To train the model, run the following command:
>python train_lindyhop_multiple.py

## Evaluation
To evaluate the pretrained model on the test subjects, run the following command:
>python eval_lindyhop_multiple.py

Results will be saved in the ‘checkpoint’ folder.
