# Active Listener: Continuous Generation Of Listener’s Head Motion Response In Dyadic Interactions
[![](https://img.shields.io/github/stars/Y-debug-sys/Diffusion-TS.svg)](https://github.com/bigzen/Active-Listener/stargazers)
[![](https://img.shields.io/github/forks/Y-debug-sys/Diffusion-TS.svg)](https://github.com/bigzen/Active-Listener/network) 
[![](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/bigzen/Active-Listener/blob/main/LICENSE) 
<img src="https://img.shields.io/badge/python-3.10-blue">
<img src="https://img.shields.io/badge/pytorch-2.4-orange">
<img src="https://img.shields.io/badge/cuda-12.1-orange"><br>
<img src="https://img.shields.io/badge/numpy-1.24-green">
<img src="https://img.shields.io/badge/pickle-0.7-green">
<img src="https://img.shields.io/badge/librosa-0.10-green">
<img src="https://img.shields.io/badge/audiofile-1.3-green">
<img src="https://img.shields.io/badge/transformers-4.33-green">
<img src="https://img.shields.io/badge/opensmile-2.4-green"><br>
<img src="https://img.shields.io/badge/scipy-1.10-green">
<img src="https://img.shields.io/badge/torchaudio-2.4-green">

> **Abstract:** A key component of dyadic spoken interactions is the contextually relevant non-verbal gestures, such as head movements that reflect a listener's response to the interlocutor's speech. Although significant progress has been made in the context of generating co-speech gestures, generating listener's response has remained a challenge. We introduce the task of generating continuous head motion response of a listener in response to the speaker's speech in real time. To this end, we propose a graph-based end-to-end crossmodal model that takes interlocutor's speech audio as input and directly generates head pose angles (roll, pitch, yaw) of the listener in real time. Different from previous work, our approach is completely data-driven, does not require manual annotations or oversimplify head motion to merely nods and shakes. Extensive evaluation on the dyadic interaction sessions on the IEMOCAP dataset shows that our model produces a low overall error (4.5 degrees) and a high frame rate, thereby indicating its deployability in real-world human-robot interaction systems.


<p align="center">
  <img src="model.svg" alt="">
  <br>
  <b>Figure 1</b>: Overall Architecture of Diffusion-TS.
</p>

## Dataset Preparation

The IEMOCAP dataset can be obtained from [USC](https://sail.usc.edu/iemocap/iemocap_release.htm). Once obtained store the dataset to the folder `./IEMOCAP_full_release` in the same parent directory as our repository. Copy the files inside the data folder to the Dataset directory.


## Running the Code

 The code requires conda3 (or miniconda3), and one CUDA capable GPU. The instructions below guide you regarding running the codes in this repository. 

### Environment & Libraries

The full libraries list is provided as a `requirements.txt` in this repo. Please create a virtual environment with `conda` or `venv` and run

~~~bash
(myenv) $ pip install -r requirements.txt
~~~

### Training & Sampling

For training, you can reproduce the experimental results of all benchmarks by runing

~~~bash
(myenv) $ python main.py --name {name} --config_file {config.yaml} --gpu 0 --train
~~~

**Note:** We also provided the corresponding `.yml` files (only stocks, sines, mujoco, etth, energy and fmri) under the folder `./Config` where all possible option can be altered. You may need to change some parameters in the model for different scenarios. For example, we use the whole data to train model for unconditional evaluation, then *training_ratio* is set to 1 by default. As for conditional generation, we need to divide data set thus it should be changed to a value < 1. 

While training, the script will save check points to the *results* folder after a fixed number of epochs. Once trained, please use the saved model for sampling by running
