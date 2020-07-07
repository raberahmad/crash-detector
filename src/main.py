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
import sys


network = TCNN()
network.prepare_data()


model = network.load_model("alexnet-test-cropped")
network.test_accuracy(model)
visual = Visual(network)




if len(sys.argv) >= 2:
    
    param = sys.argv[1]

    if param == "demo1":
        visual.folder_images('/Users/rah30032/Desktop/demo1/*.j*')
        print("end of folder")
        visual.folder_images('/Users/rah30032/Desktop/demo1.5/147.j*')
        visual.folder_images('/Users/rah30032/Desktop/demo1.5/147copy.j*')
        visual.folder_images('/Users/rah30032/Desktop/demo1.5/145.j*')
        visual.folder_images('/Users/rah30032/Desktop/demo1.5/145copy.j*')
    if param == "demo2":
        video_path = '/Users/rah30032/Desktop/demo2.mp4'
        visual.play_video(video_path)
    if param == "demo3":
        video_path = '/Users/rah30032/Desktop/demo3.mp4'
        visual.play_video(video_path)
    if param == "demo4":
        visual.folder_images('/Users/rah30032/Desktop/demo4/*.j*')
    if param == "demo5":
        video_path = '/Users/rah30032/Desktop/demo5.mov'
        visual.play_video(video_path)


    print(sys.argv[1])
else:
    print("No parameter ")
