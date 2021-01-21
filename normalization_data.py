
import argparse
import numpy as np
from imutils.paths import list_files
from pydub import AudioSegment
import os
import librosa
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="Path to folder of dataset, path_to_data = './data/'")
args = vars(ap.parse_args())

def normalization(path_to_data):
    """Load all wavs file and split 2s/file
    ex: path_to_data = './data/'

    Args:
        path_to_data (string): path to data folder

    Returns:
        No return
    """
    
    try:
        #list all folder
        paths = list_files(path_to_data)
        folders = set()
        for path_ in paths:
            folder_name = path_.split('/')[-2]
            folders.add(folder_name)
        folders = list(folders)

        #data normalization, 2s/wav file
        for folder in folders:
            paths = list_files(path_to_data + folder + '/')
            audios = AudioSegment.empty()

            #read wav file and remove it
            for path_ in paths:
                wav = AudioSegment.from_wav(path_)
                audios += wav
                os.remove(path_)

            L = len(audios)
            N = int(L/2000)

            for i in range(N):
                wav = audios[i*2000:(i+1)*2000]
                wav.export(path_to_data + folder + '/{}.wav'.format(i), format='wav')
            
            #destroy
            del audios; del L; del N;
        
    except Exception as e:
        print('Error: ', e)


def mfcc_feature_extraction(file_name):
    """MFCC Feature Extraction

    Args:
        file_name (string): wav file name

    Returns:
        [2D array]: MFCC features
    """
    max_pad_len = 100
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error encountered: ", e)
        return None 
     
    return mfccs


def convert_mfcc2img(mfcc):
    """Convert MFCC features to image

    Args:
        mfcc (2D array): mfcc features

    Returns:
        [2D array]: Image corresponding to MFCC features
    """
    try:
      mfcc = np.array(mfcc)
      
      MAX=np.max(mfcc);MIN=np.min(mfcc)
      
      #new value domain
      NEW_MAX=255; NEW_MIN=0
      
      img_mfcc = (mfcc-MIN)/(MAX-MIN) * (NEW_MAX-NEW_MIN)

      img_mfcc =img_mfcc[:,0:80]

    except Exception as e:
      print("Error encountered: ", e)
      return None;

    return img_mfcc;


def create_data(path_to_data):
    """Create dataset (contain mfcc feature) for training

    Args:
        path_to_data ([string]): path to input folder contain data
    """
    
    if not os.path.exists('mfcc'):
        os.mkdir('mfcc')
    
    #list all folder
    paths = list_files(path_to_data)
    
    folders = set()
    for path_ in paths:
        folder_name = path_.split('/')[-2]
        folders.add(folder_name)
    folders = list(folders)
    
    #create sub folder in img_data
    os.chdir('mfcc')
    for folder in folders:
        if not os.path.exists(folder):
            os.mkdir(folder)     
    os.chdir('..')
    
    paths = list_files(path_to_data)
    #convert to img_data
    for i, path_ in enumerate(paths):
        mfcc = mfcc_feature_extraction(path_)
        mfcc_img = convert_mfcc2img(mfcc)
        
        folder = path_.split('/')[-2]
        
        new_path = './mfcc/' + folder + '/{}.jpg'.format(i);
        
        cv2.imwrite(new_path, mfcc_img)
        
if __name__=='__main__':
    #input
    path_to_data = args['input']
    
    #normalization data
    normalization(path_to_data);
    
    #create dataset
    create_data(path_to_data)
    
    print('Done.')