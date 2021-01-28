# Genetic VoiceNet [![CircleCI](https://circleci.com/gh/faustomorales/keras-ocr.svg?style=shield)](https://github.com/nguyentruonglau) [![Documentation Status](https://readthedocs.org/projects/keras-ocr/badge/?version=latest)](https://github.com/nguyentruonglau)

All codes assume running from root directory. Please update the sys path at the beginning of the codes before running.
> [Evolutionary Neural Architecture Search For Vietnamese Speaker Recognition]
>
> Nguyễn Trường Lâu - Student at University of Information Technology (UIT)
>

![overview](https://github.com/nguyentruonglau/VoiceNet/blob/main/img/cnn_architecture.png "Model Architecture")

## Over View
``` 
VoiceNet - convolutional neural network architecture found by nondominated sorting genetic algorithm ii with code: 0-10 - 1-01-001 - 0-00. Two goals are optimized: accuracy and cost calculation
```

## Requirements
```
python == 3.6.9 #mandatory

glob2==0.7
h5py==2.10.0
imutils==0.5.3
Keras==2.4.3 #mandatory
librosa==0.6.3
matplotlib==3.2.2
numpy==1.19.5
opencv-contrib-python==4.1.2.30
opencv-python==4.1.2.30
pandas==1.1.5
Pillow==7.0.0
prefetch-generator==1.0.1
pydub==0.24.1
scikit-image==0.16.2
scipy==1.4.1
sklearn==0.0
spacy==2.2.4
tensorboard==2.4.0
tensorflow==2.4.0 #mandatory

pip install -r requirements.txt
```

## Sample pretrained models
``` 
model/voicenet.hdf5
```
## Setup
``` 
To setup:
  >>> pip install voicenet

To use:
  >>> from voicenet import VoiceNet
  >>> model = VoiceNet(input_shape=(40,80,1), classes=40)
  >>> model.summary()
```


## Dataset
``` 
    data--
        --Speaker One
          voice_1.wav
          voice_2.wav
        --Speaker Two
          voice_1.wav
          voice_2.wav
```

## Normalization data
``` 
python normalization_data.py --cfg=cfg/config.cfg
```

## Training
``` 
python train.py -batch_size 32 -num_class 40 -epochs 100
```

## Citations
If you find the code useful for your research, please consider citing our works
``` 
@article{voicenet,
  title={Evolutionary Neural Architecture Search For Vietnamese Speaker Recognition},
  author={Nguyễn Trường Lâu - Student at University of Information Technology (UIT)},
  booktitle={NAS},
  year={2020}
}
```
