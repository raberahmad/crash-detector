from tcnn import TCNN
from visualiser import Visual
import pafy
from PIL import Image
import numpy as np
import glob
import cv2
import pafy
import pymsgbox
import time

network = TCNN()
network.prepare_data()


model = network.load_model("alexnet-test-cropped")
network.test_accuracy(model)
visual = Visual(network)

#1
# video_path = '/Users/rah30032/Desktop/videos/mov15.mp4'
# visual.play_video(video_path)

#2
# visual.folder_images('/Users/rah30032/crash-detector/predict_folders/*.jpeg')

#3
# video_path = '/Users/rah30032/Desktop/mov1523.mp4'
# visual.play_video(video_path)
 
#4
# visual.folder_images('/Users/rah30032/crash-detector/cctv-real/*.jpeg')

#5
# visual.folder_images('/Users/rah30032/crash-detector/cctv-cropped/*.jpeg')

#6
video_path = '/Users/rah30032/Desktop/demo2.mov'
visual.play_video(video_path)