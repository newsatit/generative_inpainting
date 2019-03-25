import os
import shutil
import numpy as np

source = "base_meme_images"
train_dir = "training_data/memegenerator/training"
valid_dir = "training_data/memegenerator/validation"

files = os.listdir(source)
files = (file for file in files if file.endswith(".jpg"))

for f in files:
    if np.random.rand(1) < 0.2:
        shutil.copy(source + '/'+ f, valid_dir + '/'+ f)
    else:
        shutil.copy(source + '/'+ f, train_dir + '/'+ f)
        