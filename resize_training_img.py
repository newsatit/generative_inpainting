#!/usr/bin/python
from PIL import Image
import os, sys
import neuralgym as ng

path = "./training_data/memegenerator/training/"
dirs = os.listdir(path)
config = ng.Config('inpaint.yml')
img_size = (config.IMG_SHAPES[0], config.IMG_SHAPES[1])

print(img_size)

for item in dirs:
    if os.path.isfile(path + item) and item != '.DS_Store':
        im = Image.open(path + item)
        imResize = im.resize((256,256), Image.ANTIALIAS)
        imResize.save(path + item, 'JPEG')




