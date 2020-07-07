import cv2
import numpy as np

class DetectionCars:

    def __init__(self):
        #self.cap = cv2.VideoCapture('videos/video.avi')
        self.cap = cv2.VideoCapture('https://stream.inmoves.nl:1935/rtplive/IB_28.stream/playlist.m3u8')
        self.w = 320
        self.h = 240
        self.cap.set(3, self.w)
        self.cap.set(4, self.h)
        self.count1 = 0
        self.count2 = 0
        ret, frames = self.cap.read()
        self.height = np.size(frames, 0)
        self.width = np.size(frames, 1)
        self.car_cascade = cv2.CascadeClassifier('cars3.xml')


    def run(self):
        self.CoorYEntranceLine = int((self.height / 2) + -100)
        self.CoorYExitLine = int((self.height / 2) + -50)



        while self.cap.isOpened():
            ret, frames = self.cap.read()
            #cv2.line(frames, (0, self.CoorYEntranceLine), (self.width, self.CoorYEntranceLine), (255, 0, 0), 2)
            #cv2.line(frames, (0, self.CoorYExitLine), (self.width, self.CoorYExitLine), (0, 0, 255), 2)
            #frames = cv2.resize(frames, (self.w, self.h), interpolation=cv2.INTER_CUBIC)

            if ret:
                gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                cars = self.car_cascade.detectMultiScale(gray, 1.1, 5,)

                self.draw(frames, cars)



                cv2.imshow('video', frames)
                # cv2.imwrite("/Users/rah30032/Documents/afstudeerstage/opencv-example/saved/"+str(count)+".jpeg", frm)

            if cv2.waitKey(33) is 32:
                break




    def draw(self, frame, cars):
        i = 0
        for car_id, (x, y, w, h) in enumerate(cars):

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
            cv2.putText(frame, str(car_id), (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX,
                        .7, (0, 255, 255), 1, cv2.LINE_AA)

            CoordXCentroid = int((x + x + w) / 2)
            CoordYCentroid = int((y + y + h) / 2)
            ObjectCentroid = (CoordXCentroid, CoordYCentroid)
            cv2.circle(frame, ObjectCentroid, 1, (255, 255, 255), 6)

            if (self.checkCrossingEnterLine(CoordYCentroid, self.CoorYEntranceLine, self.CoorYExitLine)):
                self.count1 += 1

            if (self.checkCrossingExitLine(CoordYCentroid, self.CoorYEntranceLine, self.CoorYExitLine)):
                self.count2 += 1

            i+=1
            print(i)
            print('cars Enter: ', self.count1, " cars Exit: ", self.count2)


    def checkCrossingEnterLine(self,y, cYEnterLine, cYExitLine):
        AbsDistance = abs(y - cYEnterLine)

        if ((AbsDistance <= 2) and (y < cYExitLine)):
            return 1
        else:
            return 0

    def checkCrossingExitLine(self, y, cyEnterLine, CoorYExitLine):
            AbsDistance = abs(y - CoorYExitLine)

            if ((AbsDistance <= 2) and (y > cyEnterLine)):
                return 1
            else:
                return 0



    def __del__(self):
        cv2.destroyAllWindows()
        print('destructor works')