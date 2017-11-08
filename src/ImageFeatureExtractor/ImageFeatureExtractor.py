from PIL import Image, ImageStat
import re # Regular expressions
from time import sleep,ctime # To prevent overwhelming the server between connections
from collections import Counter # Keep track of our term counts
import sys, getopt, glob
import os
import cv2
from datetime import datetime, timedelta
import numpy as np
from numpy.linalg import norm
import signal
import random


class ImageFeatureExtractor:
    def __init__(self, imageFileName, facecascxml_path, eyecascxml_path):
        self.feature_vector = []
        self.inputimagefile = imageFileName
        self.inputfacecascxml = facecascxml_path
        self.inputeyecasxml = eyecascxml_path

    def initialize(self):
        if not os.path.isfile(self.inputimagefile):
            self.quit("Input image file doesn't exist or it's not a file")
        if not os.path.isfile(self.inputfacecascxml):
            self.quit("Input face haarcascade xml doesn't exist or it's not a file")
        if not os.path.isfile(self.inputeyecasxml):
            self.quit("Input eye haarcascade xml doesn't exist or it's not a file")

    def extract_img_feataure(self):
        face_hue = []
        face_brightness = []
        eye_signal = []
        face_ratio =[]
        face_SNR = []
        face_num = 0
        img_face_hue = 0.0
        img_face_brightness=0.0
        img_face_worstSNR=0.0
        img_eye_signal =0.0

        facehc_xml = self.inputfacecascxml
        eyehc_xml = self.inputeyecasxml
        img = self.inputimagefile
        face_cascade = cv2.CascadeClassifier(facehc_xml)
        eye_cascade = cv2.CascadeClassifier(eyehc_xml)
        image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        mu, std = cv2.meanStdDev(gray_laplacian)
        img_sharpness=np.log2(float(std.item()*std.item()))
        img_contrast = float(np.amax(gray)-np.amin(gray))/float(np.amax(gray)+np.amin(gray))
        img_weight, img_height = gray.shape
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        face_num = len(faces)
        print "Found {0} faces!".format(len(faces))

        # Draw a rectangle around the faces
        if len(faces):
            for (x, y, w, h) in faces:
                eye_signal_face = []
                face_roi = image[y:y+h, x:x+w, :]
                face_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
                hue = face_hsv[:, :, 0]
                brightness = face_hsv[:, :, 2]
                face_gray_roi = gray[y:y+h, x:x+w]
                face_ratio_temp = np.log2(float(h*w)/float(img_height*img_weight))
                eyes = eye_cascade.detectMultiScale(face_gray_roi)
                if len(eyes):
                    for (ex, ey, ew, eh) in eyes:
                        eye_signal_face.append(np.average(gray[ey:ey+eh, ex:ex+ew]))
                        hue[ey:ey+eh, ex:ex+ew] = -1
                        brightness[ey:ey + eh, ex:ex + ew] = -1
                    eye_signal.append(np.average(eye_signal_face))
                hue_mask = np.ma.masked_equal(hue, -1).compressed()
                brightness_mask = np.ma.masked_equal(brightness, -1).compressed()
                face_hue.append(np.average(hue_mask))
                face_brightness.append(np.average(brightness_mask))
                face_SNR.append(20*np.log10(np.average(brightness_mask)/np.std(brightness_mask)))
                face_ratio.append(face_ratio_temp)

            img_face_hue = np.average(face_hue)
            img_face_brightness = np.average(face_brightness)
            img_face_worstSNR = np.amin(face_SNR)
            img_face_ratio = np.average(face_ratio)
            img_eye_signal = np.average(eye_signal)

        self.feature_vector = [face_num, img_face_brightness, img_face_hue, img_face_ratio, img_face_worstSNR, img_eye_signal, img_contrast, img_sharpness]

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                               Main Method                                    """
"""                                                                              """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
    testimage = '/Users/lingouyang/GoogleDrive/sjsu/CS256/cs256_Project/AtlanticCity.jpg'
    face_hcxml = '/Users/lingouyang/GoogleDrive/sjsu/CS256/cs256_Project/haarcascade_frontalface_default.xml'
    eye_hcxml = '/Users/lingouyang/GoogleDrive/sjsu/CS256/cs256_Project/haarcascade_eye.xml'
    im_extractor = ImageFeatureExtractor(testimage, face_hcxml, eye_hcxml)
    im_extractor.initialize()
    im_extractor.extract_img_feataure()
    print im_extractor.feature_vector


