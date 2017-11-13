import os
import sys

import cv2
import numpy as np


class ImageFeatureExtractor:



	EYE_XML = os.path.join(os.path.dirname(__file__), 'xml', 'HAAR_EYE.xml')
	FACE_XML = os.path.join(os.path.dirname(__file__), 'xml', 'HAAR_FACE.xml')
	def __init__(self, imageFileName, facecascxml_path, eyecascxml_path):

		self.feature_vector = []
		self.inputimagefile = imageFileName
		self.inputfacecascxml = os.path.abspath(facecascxml_path)
		self.inputeyecasxml = os.path.abspath(eyecascxml_path)

	def initialize(self):

		if not os.path.isfile(self.inputimagefile):
			print("Input image file doesn't exist or it's not a file")
			sys.exit(2)
		if not os.path.isfile(self.inputfacecascxml):
			print("Input face haarcascade xml doesn't exist or it's not a file")
			sys.exit(2)
		if not os.path.isfile(self.inputeyecasxml):
			print("Input eye haarcascade xml doesn't exist or it's not a file")
			sys.exit(2)

	def extract(self):
		face_hue = []
		face_brightness = []
		eye_signal = []
		face_ratio = []
		face_SNR = []
		face_num = 0
		img_face_hue = 0.0
		img_face_brightness = 0.0
		img_face_worstSNR = 0.0
		img_eye_signal = 0.0
		img_contrast = 0.0

		facehc_xml = self.inputfacecascxml
		eyehc_xml = self.inputeyecasxml
		img = self.inputimagefile
		face_cascade = cv2.CascadeClassifier(ImageFeatureExtractor.FACE_XML)
		eye_cascade = cv2.CascadeClassifier(ImageFeatureExtractor.FACE_XML)
		image = cv2.imread(img)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		gray_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
		mu, std = cv2.meanStdDev(gray_laplacian)
		img_sharpness = np.log2(float(std.item() * std.item()))

		sum_contrast = float(np.amax(gray)) + float(np.amin(gray))
		diff_contrast = float(np.amax(gray)) - float(np.amin(gray))
		img_contrast = diff_contrast / sum_contrast

		img_weight, img_height = gray.shape
		print self.inputimagefile
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		face_num = len(faces)
		img_face_ratio = 0.0
		# print "Found {0} faces!".format(len(faces))

		# Draw a rectangle around the faces
		if len(faces):
			for (x, y, w, h) in faces:
				eye_signal_face = []
				face_roi = image[y:y + h, x:x + w, :]
				face_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
				hue = face_hsv[:, :, 0]
				brightness = face_hsv[:, :, 2]
				face_gray_roi = gray[y:y + h, x:x + w]
				face_ratio_temp = np.log2(float(h * w) / float(img_height * img_weight))
				eyes = eye_cascade.detectMultiScale(face_gray_roi)
				if len(eyes):
					for (ex, ey, ew, eh) in eyes:
						eye_signal_face.append(np.average(gray[ey:ey + eh, ex:ex + ew]))
						hue[ey:ey + eh, ex:ex + ew] = -1
						brightness[ey:ey + eh, ex:ex + ew] = -1
					eye_signal.append(np.average(eye_signal_face))
					img_eye_signal = np.average(eye_signal)

				hue_mask = np.ma.masked_equal(hue, -1).compressed()
				brightness_mask = np.ma.masked_equal(brightness, -1).compressed()
				face_hue.append(np.average(hue_mask))
				face_brightness.append(np.average(brightness_mask))
				face_SNR.append(20 * np.log10(np.average(brightness_mask) / np.std(brightness_mask)))
				face_ratio.append(face_ratio_temp)

			img_face_hue = np.average(face_hue)
			img_face_brightness = np.average(face_brightness)
			img_face_worstSNR = np.amin(face_SNR)
			img_face_ratio = np.average(face_ratio)

		self.feature_vector = [face_num, img_face_brightness, img_face_hue, img_face_ratio, img_face_worstSNR,
		                       img_eye_signal, img_contrast, img_sharpness]
		return self.feature_vector
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                               Main Method                                    """
"""                                                                              """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	testimage = '../data/training/ling/1/JapanVisit.JPG'
	face_hcxml = './xml/HAAR_FACE.xml'
	eye_hcxml = './xml//HAAR_EYE.xml'
	im_extractor = ImageFeatureExtractor(testimage, face_hcxml, eye_hcxml)
	im_extractor.initialize()
	im_extractor.extract()
	print im_extractor.feature_vector
