import pafy
from PIL import Image
import numpy as np
import glob
import cv2
import pafy
import time


class Visual:

    def __init__(self, network_to_visual):
        self.network = network_to_visual
     
        
    def play_video(self, videostream):
        stream = videostream
        
        frame_skipper_counter = 0
        if stream[6: 13] == '//youtu':
            video = pafy.new(stream)
            best = video.getbest(preftype="mp4")

            self.cap = cv2.VideoCapture(best.url)
        else:
            self.cap = cv2.VideoCapture(stream)

        self.w = 320
        self.h = 240
        self.cap.set(3, self.w)
        self.cap.set(4, self.h)
        
        ret, frames = self.cap.read()
        print(ret, frames)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            time_start = time.time()
            if ret:
                frame_skipper_counter = frame_skipper_counter + 1
                # if frame_skipper_counter % 5 == 0:
                self.visual_to_prediction(frame)
                cv2.imshow('video', frame)
                frame_skipper_counter = 0
            
            while True:
                if cv2.waitKey(33) == 32:
                    break


    def folder_images(self, imagefolder):
        folder = imagefolder 
        
        image_list = []
        for filename in glob.glob(folder): 
            im=Image.open(filename)
            image_list.append(im)


        for image in image_list:
            topredict = self.process_img(image, False)


            start = time.time()


            probs, classes = self.network.predict_class(topredict)
            end = time.time()
            
            print("Prediction time: {}".format(end - start))
            print("Accuracy: {} | Result: {}".format(probs[0], classes[0]))

            image = np.array(image)
            cv2.imshow('video', image)


            if classes[0] == "pos":
                print("\007")
                #pymsgbox.alert('Accidant on screen!', 'WARNING')
        
            while True:
                if cv2.waitKey(33) == 32:
                    break



        


    def visual_to_prediction(self, img_to_network):
        start = time.time()
        topredict = self.process_img(img_to_network, True)


        end = time.time()
        probs, classes = self.network.predict_class(topredict)
        print("Prediction time: {}".format(end - start))
        print("Accuracy: {} | Result: {}".format(probs[0], classes[0]))


    
    def process_img(self, image, isVideo):
       #pil_image = Image.fromarray(img)
        # pil_image = img
        
        
        if isVideo:
            pil_image = Image.fromarray(image)
        else:
            pil_image = image




        
    
        pil_image.thumbnail((500, 500))
            
        # Crop 
        left_margin = (pil_image.width-224)/2
        bottom_margin = (pil_image.height-224)/2
        right_margin = left_margin + 224
        top_margin = bottom_margin + 224

        pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))

        # Normalize
        np_image = np.array(pil_image)/255
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - mean) / std

        # PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array
        # Color channel needs to be first; retain the order of the other two dimensions.
        np_image = np_image.transpose((2, 0, 1))

        return np_image


