from tqdm import tqdm
import numpy as np
from numpy import fliplr, flipud
import numpy as np
from skimage.io import imread, imsave
import image_slicer
from skimage.transform import rotate

import dataset


data = dataset.Dataset("/Users/rah30032/PycharmProjects/crash-detector/dataset-beamng/dataset-ongevallen.xlsx")
data_excel = data.importData()

train_image = []
aug_image = []
aug_collision_variables = []

for index, row in tqdm(data_excel.iterrows()):
    print(row['images_name'], row['collision'])




    try:

        img_path = "/Users/rah30032/PycharmProjects/crash-detector/dataset-beamng/train/" + row['images_name']

        tiles = image_slicer.slice(img_path, 6, save=False)
        image_slicer.save_tiles(tiles, directory='/Users/rah30032/PycharmProjects/crash-detector/augmented/', prefix='slice'+(row['images_name'])[:-5] , format='jpeg')
        img = imread(img_path)

        img = img.astype('uint8')

        # images are being rotated and flipped
        img_rot_90 = rotate(img, 90)
        img_rot_180 = rotate(img, 180)
        img_rot_270 = rotate(img, 270)
        img_flip_h = fliplr(img)
        img_flip_v = flipud(img)


        if row['collision'] == 1:
            imsave("augmented/pos/img_rot_90"+str(index)+".jpeg", img_rot_90)
            imsave("augmented/pos/img_rot_180" + str(index) + ".jpeg", img_rot_180)
            imsave("augmented/pos/img_rot_270" + str(index) + ".jpeg", img_rot_270)
            imsave("augmented/pos/img_flip_h" + str(index) + ".jpeg", img_flip_h)
            imsave("augmented/pos/img_flip_v" + str(index) + ".jpeg", img_flip_v)
        elif row["collision"] == 0:
            imsave("augmented/neg/img_rot_90" + str(index) + ".jpeg", img_rot_90)
            imsave("augmented/neg/img_rot_180" + str(index) + ".jpeg", img_rot_180)
            imsave("augmented/neg/img_rot_270" + str(index) + ".jpeg", img_rot_270)
            imsave("augmented/neg/img_flip_h" + str(index) + ".jpeg", img_flip_h)
            imsave("augmented/neg/img_flip_v" + str(index) + ".jpeg", img_flip_v)
    except:
        print("File not found: "+img_path)
        continue




