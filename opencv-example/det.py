import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox


cap = cv2.VideoCapture('https://stream.inmoves.nl:1935/rtplive/IB_231.stream/playlist.m3u8')

ret, frames = cap.read()


bbox, label, conf = cv.detect_common_objects(frames)
output_image = draw_bbox(frames, bbox, label, conf)
plt.imshow(output_image)
plt.show()
print('Number of cars in the image is '+ str(label.count('car')))