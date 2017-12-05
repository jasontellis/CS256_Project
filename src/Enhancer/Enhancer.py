# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# import copy
# # from PIL import Image, ImageStat
# # import matplotlib.mlab as mlab
# # import matplotlib.pyplot as plt
# import datetime
# import cv2
# from skimage.color import rgb2lab, lab2lch, lab2rgb, rgb2yuv, yuv2rgb
# import numpy as np
# from ImageFeatureExtractor import ImageFeatureExtractor
#
# class Enhancer:
#
# 	def enhance(self, ref_feature_vector = [2, 0.5, 19, 20, 7, 4, 7.05, 10], enhance = True, featureVector = None, imageFileName = None):
#
# 		bool_entropy = False
# 		bool_sharpen = False
# 		bool_denoise = False
# 		bool_cc = False
# 		self.feature_vector = featureVector
# 		self.inputimagefile = imageFileName
#
# 		cur_feature_vector = self.feature_vector
# 		basePath = os.path.dirname(__file__)
# 		# print 'Base', basePath
# 		parentPath = os.path.abspath(os.path.join(basePath, ".."))
# 		enhanceimg_Dir = os.path.abspath(os.path.join(parentPath, 'data', 'enhance'))
# 		# print ehDir
#
# 		# enhanceimg_Dir="../data/enhance/"
# 		if not os.path.exists(enhanceimg_Dir):
# 			os.mkdir(enhanceimg_Dir)
# 		enhanceimg_Name = "".join([enhanceimg_Dir, "/",
# 		                           self.inputimagefile.split('/')[-1].split('.')[0] + '{:-%Y-%m-%d_%H-%M-%S}'.format(
# 			                           datetime.datetime.now()) + '.jpg'])
# 		img_global_entropy = cur_feature_vector[-2]
# 		ref_global_entropy = ref_feature_vector[-2]
# 		if img_global_entropy < ref_global_entropy:
# 			cliplimit_value = max(min(2, float(ref_global_entropy - img_global_entropy) * 10), 1)
#
# 			# ling
# 			self.enhance_constrast_brightness(cliplimit_th = cliplimit_value)
# 			bool_entropy = True
# 		# cv2.startWindowThread()
# 		# cv2.namedWindow("Contrasted",cv2.WINDOW_NORMAL)
# 		# cv2.imshow("Contrasted",self.image)
#
# 		if cur_feature_vector[0] < 1:
# 			print(
# 			"no face detected in this bad image so enhancement would focus on non-face feature such as contrast and brightness")
# 			# if os.path.isfile(enhanceimg_Name):
# 			print enhanceimg_Name, "is generated and saved"
# 			cv2.imwrite(enhanceimg_Name, self.image)
# 			# else:
# 			print "Enhaced images is saved as", enhanceimg_Name
# 		else:
# 			img_face_sharpness = cur_feature_vector[4]
# 			ref_face_sharpness = ref_feature_vector[4]
#
# 			if img_face_sharpness < ref_face_sharpness:
# 				self.img_sharpen()
# 				bool_sharpen = True
# 				# cv2.startWindowThread()
# 				# cv2.namedWindow("Sharpened",cv2.WINDOW_NORMAL)
# 				# cv2.imshow("Sharpened",self.image)
# 				print (
# 				"current image sharpness:" + repr(img_face_sharpness) + " < " + "reference image sharpness:" + repr(
# 					ref_face_sharpness))
# 				print("image sharpening is performed due to blurred image")
# 			# cv2.waitKey(10000)
#
# 			img_face_worstSNR = cur_feature_vector[5]
# 			ref_face_worstSNR = ref_feature_vector[5]
# 			if img_face_worstSNR < ref_face_worstSNR:
# 				self.skin_noise_reduction()
# 				bool_denoise = True
# 				print ("current face SNR:" + repr(img_face_worstSNR) + " < " + "reference image SNR:" + repr(
# 					ref_face_worstSNR))
# 				print("image denoise is performed due to noisy face")
#
# 			img_face_color_pr = cur_feature_vector[1]
# 			ref_face_color_pr = ref_feature_vector[1]
# 			ref_face_skin_a = ref_feature_vector[2]
# 			ref_face_skin_b = ref_feature_vector[3]
#
# 			if img_face_color_pr < ref_face_color_pr:
# 				self.skin_color_correction(ref_face_skin_a, ref_face_skin_b)
# 				bool_cc = True
# 				print ("current face color accuracy:" + repr(
# 					img_face_color_pr) + " < " + "reference face color accuracy:" + repr(ref_face_color_pr))
# 				print("face color correction is performed due to inaccurate face skin tone color")
#
# 			if bool_entropy == True or bool_denoise == True or bool_sharpen == True or bool_cc == True:
# 				# cv2.startWindowThread()
# 				# cv2.namedWindow("Enhanced",cv2.WINDOW_NORMAL)
# 				# cv2.imshow("Enhanced",self.image)
# 				# cv2.imwrite(enhanceimg_Name ,self.image)
# 				if os.path.isfile(enhanceimg_Name):
# 					print enhanceimg_Name, "is generated and saved"
# 			else:
# 				print "No need to perform image enhancement on this image, please try another one"
#
# 	def enhance_constrast_brightness(self, gridsize = 8, cliplimit_th = 5):
# 		faces = self.face_ROI
# 		img = self.image
# 		ori_img = copy.deepcopy(img)
# 		# cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
# 		# cv2.imshow("Original", ori_img)
# 		if len(faces) > 0:
# 			for (x, y, w, h) in faces:
# 				face_roi = copy.deepcopy(img[y:y + h, x:x + w, :])
# 			# cv2.rectangle(face_roi,(x,y),(x+w,y+h),(0,255,255),2)
# 			# cv2.startWindowThread()
# 			# cv2.namedWindow("face_low_contast",cv2.WINDOW_NORMAL)
# 			# cv2.imshow("face_low_contast",face_roi)
# 			self.feature_vector[0] = len(faces)
# 		# cv2.waitKey(0)
# 		img_lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
# 		img_lch = lab2lch(img_lab)
# 		lab_planes = cv2.split(img_lab)
# 		clahe = cv2.createCLAHE(clipLimit = cliplimit_th, tileGridSize = (gridsize, gridsize))
# 		lab_planes[0] = clahe.apply(lab_planes[0])
# 		lab = cv2.merge(lab_planes)
# 		img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
# 		img_enhanced2 = cv2.bilateralFilter(img_enhanced, 9, 10, 10)
# 		gray_enhanced = cv2.cvtColor(img_enhanced2, cv2.COLOR_BGR2GRAY)
# 		faces = self.face_cascade.detectMultiScale(gray_enhanced, 1.3, 5)
# 		if len(faces) > 0:
# 			for (x, y, w, h) in faces:
# 				face_roi_enhance = copy.deepcopy(img_enhanced2[y:y + h, x:x + w, :])
# 			# cv2.rectangle(face_roi_enhance,(x,y),(x+w,y+h),(0,255,255),2)
# 			# cv2.startWindowThread()
# 			# cv2.namedWindow("face_highcontrast",cv2.WINDOW_NORMAL)
# 			# cv2.imshow("face_highcontrast", face_roi_enhance)
# 			self.face_ROI = faces
#
# 		# cv2.namedWindow("contrast",cv2.WINDOW_NORMAL)
# 		# cv2.imshow("contrast",img_enhanced2)
# 		# cv2.waitKey(5000)
#
# 		# img=self.image
#
#
# 		# cv2.waitKey(0)
# 		# cv2.imwrite("../data/training/mugdha/enhance_image.jpg",bgr)
#
#
# 		# self.face_ROI=faces
# 		self.image = img_enhanced2
#
# 	def skin_noise_reduction(self):
# 		faces = self.face_ROI
# 		img = self.image
# 		for (x, y, w, h) in faces:
# 			face_roi = copy.deepcopy(img[y:y + h, x:x + w, :])
#
# 			face_lab = rgb2lab(face_roi)
# 			# face_lab=
# 			face_lch = lab2lch(face_lab)
# 			face_pr = np.exp(-(np.power((face_lch[:, :, 1] - 32), 2).astype(float)) / (
# 			2 * np.power(11, 2) + np.power((face_lch[:, :, 2] - 5), 2).astype(float) / (2 * np.power(8, 2))))
# 			dim = face_pr.shape
# 			c0_width = max(1, float(dim[0]) / 50)
# 			c1_width = max(1, float(dim[1]) / 50)
# 			face_central_pr = face_pr[int(dim[0] / 2 - c0_width):int(dim[0] / 2 + c0_width),
# 			                  int(dim[1] / 2 - c1_width):int(dim[1] / 2 + c1_width)]
# 			max_denoise_color_sigma = 100 * np.mean(face_central_pr, axis = (0, 1))
# 			denoise_spatial_sigma = 100 * np.mean(face_central_pr, axis = (0, 1))
# 			# max_denoise_color_sigma=5
# 			# denoise_spatial_sigma=10
# 			face_smooth = cv2.bilateralFilter(face_roi, 9, denoise_spatial_sigma, max_denoise_color_sigma)
# 			# cv2.startWindowThread()
# 			# cv2.namedWindow("face_smooth",cv2.WINDOW_NORMAL)
# 			# cv2.imshow("face_smooth", face_smooth)
#
# 			img[y:y + h, x:x + w, :] = face_smooth
#
# 		self.image = img
#
# 	def face_sharpen(self):
# 		faces = self.face_ROI
# 		img = self.image
# 		for (x, y, w, h) in faces:
# 			face_roi = copy.deepcopy(img[y:y + h, x:x + w, :])
# 			face_yuv = cv2.cvtColor(face_roi, cv2.COLOR_RGB2YUV)
# 			# face_yuv=rgb2yuv(face_roi)
# 			# face_roi2=yuv2rgb(face_yuv)*256
# 			face_roi2 = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2RGB)
# 			# cv2.namedWindow("roi1",cv2.WINDOW_NORMAL)
# 			# cv2.imshow("roi1", face_roi2)
# 			# cv2.startWindowThread()
# 			# cv2.namedWindow("unsharpen",cv2.WINDOW_NORMAL)
# 			# cv2.imshow("unsharpen", face_roi)
#
#
#
# 			kernel = np.zeros((9, 9), np.float32)
# 			kernel[4, 4] = 2.0  # Identity, times two!
#
# 			# Create a box filter:
# 			boxFilter = np.ones((9, 9), np.float32) / 81.0
#
# 			# Subtract the two:
# 			kernel = kernel - boxFilter
#
# 			custom = cv2.filter2D(face_yuv[:, :, 0], -1, kernel)
#
# 			face_yuv[:, :, 0] = custom
# 			# face_roi_sharpen=(yuv2rgb(face_yuv))
# 			face_roi_sharpen = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2RGB)
# 			# cv2.namedWindow("Sharpen",cv2.WINDOW_NORMAL)
# 			# cv2.imshow("Sharpen", face_roi_sharpen)
# 			# cv2.waitKey(10000)
# 			img[y:y + h, x:x + w, :] = face_roi_sharpen
# 		self.image = img
#
# 	def img_sharpen(self):
# 		img = self.image
# 		img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
# 		kernel = np.zeros((9, 9), np.float32)
# 		kernel[4, 4] = 2.0  # Identity, times two!
# 		# Create a box filter:
# 		boxFilter = np.ones((9, 9), np.float32) / 81.0
# 		# Subtract the two:
# 		kernel = kernel - boxFilter
# 		custom = cv2.filter2D(img_yuv[:, :, 0], -1, kernel)
# 		img_yuv[:, :, 0] = custom
# 		# face_roi_sharpen=(yuv2rgb(face_yuv))
# 		img_sharpen = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
# 		# cv2.namedWindow("Sharpen",cv2.WINDOW_NORMAL)
# 		# cv2.imshow("Sharpen", img_sharpen)
# 		# cv2.waitKey(10000)
# 		# img[y:y + h, x:x + w, :] = face_roi_sharpen
# 		self.image = img_sharpen
#
# 	def skin_color_correction(self, ref_a, ref_b):
# 		faces = self.face_ROI
# 		img = self.image
# 		for (x, y, w, h) in faces:
# 			face_roi = copy.deepcopy(img[y:y + h, x:x + w, :])
# 			# cv2.startWindowThread()
# 			# cv2.namedWindow("face_wrong_color", cv2.WINDOW_NORMAL)
# 			# cv2.imshow("face_wrong_color", face_roi)
# 			# face_lab=cv2.cvtColor(face_roi, cv2.COLOR_RGB2LAB)
# 			face_lab = rgb2lab(face_roi)
# 			face_lch = lab2lch(face_lab)
# 			face_pr = np.exp(-(np.power((face_lch[:, :, 1] - 32), 2).astype(float)) / (
# 			2 * np.power(11, 2) + np.power((face_lch[:, :, 2] - 5), 2).astype(float) / (2 * np.power(8, 2))))
# 			a_diff = np.multiply((face_lab[:, :, 1] - ref_a), face_pr * 0.33)
# 			b_diff = np.multiply((face_lab[:, :, 2] - ref_b), face_pr * 0.33)
# 			face_lab[:, :, 1] = face_lab[:, :, 1] + a_diff
# 			face_lab[:, :, 2] = face_lab[:, :, 2] + b_diff
# 			face_skin_corrected = lab2rgb(face_lab)
# 			face_skin_corrected_new = (face_skin_corrected * 256).astype(int).clip(0, 255)
# 			# face_skin_corrected =cv2.cvtColor(face_lab, cv2.COLOR_lab2rgb)
# 			# Ling
# 			# cv2.namedWindow("Face_color_corrected",cv2.WINDOW_NORMAL)
# 			# cv2.imshow("Face_color_corrected", face_skin_corrected)
#
# 			# cv2.waitKey(30000)
# 			# Ling
#
# 			img[y:y + h, x:x + w, :] = face_skin_corrected_new
# 		# cv2.namedWindow("Whole_Image",cv2.WINDOW_NORMAL)
# 		# cv2.imshow("Whole_Image", img)
# 		# cv2.waitKey(10000)
#
# 		self.image = img
#
# if __name__ == '__main__':
# 	testimage = os.path.abspath('../data/test_1st/0/IMG_2193.jpg')
# 	feature = ImageFeatureExtractor(testimage).extract()
# 	good_feature_vector = [2, 0.5, 19, 20, 7, 8, 7.05, 10]
# 	Enhancer().enhance(good_feature_vector, True, feature, testimage)
