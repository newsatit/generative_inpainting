import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import requests



def reload(flist_path):
    with open(flist_path) as f:
        files = f.read().splitlines()
    no_empty = False    
    while not no_empty:
        no_empty = True
        for file in files:
            img_id = file[48:-4]
            
            if(not cv2.imread(file, 0) is None):
                continue
            print(img_id)
            print(flist_path, img_id, file)
            no_empty = False
            img_url = 'https://memegenerator.net/img/images/' + img_id + '.jpg'
            img_data = requests.get(img_url).content
            with open(file, 'wb') as handler:
                handler.write(img_data)

reload('./data_flist/train_shuffled.flist')
reload('./data_flist/validation_shuffled.flist')

