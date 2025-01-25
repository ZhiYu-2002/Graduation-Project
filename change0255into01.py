'''
import os
from PIL import Image
import numpy as np

if __name__ == '__main__':

    work_dir = 'E:/U-Net/data_endovis17/selfbuilder/train/selfgroundtruth1'
    file_names = os.listdir(work_dir)
    for file_name in file_names:

        file_path = os.path.join(work_dir, file_name)
        image = Image.open(file_path)
        img = np.array(image)
        img[img==255] = 1

        image = Image.fromarray(img, 'L')
        #new_name = file_name[:-4]
        #image.save(f'{new_name}.png')
        image.save(f'{file_name}')
'''

import os
from PIL import Image

path = 'E:/U-Net/data_endovis17/selfbuilder/train/selfgroundtruth1'
file_list = [os.path.join(path, f) for f in os.listdir(path)]
for file_name in file_list:
    img = Image.open(file_name)
    Img = img.convert('1')
    #b = os.path.split(file_name)[1]
    Img.save(os.path.join(path, file_name))