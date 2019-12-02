# adversarial_robustness_audio

In this work we focus on the audio domain and show that adversarial training can be applied to speech-to-text and speaker verification models. We demonstrate the feasibility of adversarial robustness in these two domains by training adversarially using a speech commands dataset. We additionally demonstrate the effectiveness of adversarial training for robustness against targeted attacks in the speaker verification domain.

## Install

This project requires Python and the following Python libraries installed:

* numpy
* PyTorch
* Scipy

## Speech-to_text

The models are run in the `audio_training_spectrogram.ipynb` and `audio_training_mfcc_cnn.ipynb` notebooks. For loading the data use `dataloader.py` and `CustomDataset.py` for spectrogram input data, or `CustomDatasetMFCC.py` to generate the PyTorch dataloader. 

## Speaker Verfication

The models for speaker verification are run in `sv_mfcc_cnn.ipynb` and `sv_spectograms.ipynb`, with targeted attacks on speaker verification is in `targeted_attacks.ipynb`. For speaker verification, use the data loader provided in the `Speaker-Verification` folder.

## Attacks

The attack functions (FGSM and PGD) are stored in `attacks.py`, and the basetrainer class for training and evaluating a model is found in `basetrainer.py`. 
he ResNet34 architecture is found in the "models" folder, and examples of audio reconstruction from spectrograms is found in `audio_reconstruction.ipynb` and from MFCC in `audio_reconstruction_MFCC.ipynb`.

