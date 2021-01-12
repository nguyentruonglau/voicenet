# Genetic VoiceNet [![CircleCI](https://circleci.com/gh/faustomorales/keras-ocr.svg?style=shield)](https://github.com/nguyentruonglau) [![Documentation Status](https://readthedocs.org/projects/keras-ocr/badge/?version=latest)](https://github.com/nguyentruonglau)

Code accompanying the paper. All codes assume running from root directory. Please update the sys path at the beginning of the codes before running.
> [Evolutionary Neural Architecture Search For Vietnamese Speaker Recognition]
>
> Nguyễn Trường Lâu - Student at University of Information Technology (UIT)
>

![overview](https://github.com/nguyentruonglau/Genetic-VoiceNet/blob/main/img/architecture.png "Model Architecture")

## Over View
``` 
This is the architecture of the model I looked for based on the Genetic Algorithm on the Famous Voice Dataset.
```

## Requirements
``` 
Python == 3.7.6, Tensorflow == 2.3.0, Keras == 2.4.3, Pydub == 0.24.1, Librosa

pip install -r requirements.txt
```

## Pretrained models
``` 
model/voicenet.hdf5
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
python normalization_data.py -i ./data/
```

## Training
``` 
python train.py -batch 64 -class 80 -epochs 100
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
