# Genetic VoiceNet
Code accompanying the paper. All codes assume running from root directory. Please update the sys path at the beginning of the codes before running.
> [Evolutionary Neural Architecture Search For Vietnamese Speaker Recognition]
>
> Nguyễn Trường Lâu - Student at University of Information Technology (UIT)
>

![overview](https://github.com/nguyen-truong-lau/genetic-voicenet/blob/main/img/model2model.png "Overview of Genetic VoiceNet")

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
        --John
          voice_1.wav
          voice_2.wav
        --Stone
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
@article{nsganet,
  title={Evolutionary Neural Architecture Search For Vietnamese Speaker Recognition},
  author={Nguyễn Trường Lâu - Student at University of Information Technology (UIT)},
  booktitle={NAS},
  year={2020}
}
```
