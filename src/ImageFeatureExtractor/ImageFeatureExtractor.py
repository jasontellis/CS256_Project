import os
import sys
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'+os.sep)))
# from PIL import Image, ImageStat
# import matplotlib.mlab as mlab
# import matplotlib.pyplot as plt
import datetime
import cv2
from skimage.color import rgb2lab, lab2lch, lab2rgb, rgb2yuv, yuv2rgb
import numpy as np


class ImageFeatureExtractor:




	def __init__(self, imageFileName, facecascxml_path="", eyecascxml_path=""):

		self.feature_vector = []
		self.face_ROI=[]
		self.image = None
		self.inputimagefile = imageFileName
		self.inputfacecascxml = os.path.abspath(facecascxml_path)
		self.inputeyecasxml = os.path.abspath(eyecascxml_path)
		self.EYE_XML = os.path.join(os.path.dirname(__file__), 'xml', 'haarcascade_eye_tree_eyeglasses.xml')
		self.FACE_XML = os.path.join(os.path.dirname(__file__), 'xml', 'HAAR_FACE.xml')
		self.enhancedDirectory = ''
		self.initialize()


	def initialize(self):

		if not os.path.isfile(self.inputimagefile):
			print("Input image file doesn't exist or it's not a file")
		
		if not os.path.isfile(self.inputfacecascxml):
			#print("Input face haarcascade xml doesn't exist or it's not a file, use default one")
			self.inputfacecascxml = self.FACE_XML
		if not os.path.isfile(self.inputeyecasxml):
			#print("Input eye haarcascade xml doesn't exist or it's not a file. Use default one")
			self.inputeyecasxml = self.EYE_XML

		self.face_cascade = cv2.CascadeClassifier(self.inputfacecascxml)
		self.eye_cascade = cv2.CascadeClassifier(self.inputeyecasxml)
		self.image = cv2.imread(self.inputimagefile)

	def extract(self):
		self.initialize()
		face_hue = []
		face_brightness = []
		face_sharpness = []
		eye_signal = []
		face_ratio = []
		face_SNR = []
		face_skin_a=[]
		face_skin_b=[]
		face_skin_h=[]
		face_skin_c=[]
		skin_pr =[]
		face_num = 0
		img_face_worstSNR = 0.0
		img_face_ratio = 0.0
		img_face_sharpness = 0.0
		img_skin_pr =0.0
		img_gray_ep = 0.0
		img_skin_a = 0
		img_skin_b = 0
		img=self.image
		facehc_xml = self.inputfacecascxml
		eyehc_xml = self.inputeyecasxml
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		self.imggray=gray
		img_gray_ep=self.calEntroy(gray)
		
		#print "image entropy:", img_gray_ep
		

		gray_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
		mu, std = cv2.meanStdDev(gray_laplacian)
		img_sharpness = np.log2(float(std.item() * std.item()))

		sum_contrast = float(np.amax(gray)) + float(np.amin(gray))
		diff_contrast = float(np.amax(gray)) - float(np.amin(gray))
		img_contrast = diff_contrast / sum_contrast

		img_weight, img_height = gray.shape
		#print self.inputimagefile
		faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
		
		face_num = len(faces)
		img_face_ratio = 0.0
		# print "Found {0} faces!".format(len(faces))

		# Draw a rectangle around the faces
		if len(faces):
			self.face_ROI = faces
			for (x, y, w, h) in faces:
				eye_signal_face = []
				face_roi = img[y:y + h, x:x + w, :]
				#cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
				face_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
				face_lab=rgb2lab(face_roi)
				#face_lab=cv2.cvtColor(face_roi, cv2.COLOR_RGB2LAB)
				face_lch=lab2lch(face_lab)
				dim=face_lch.shape
				c0_width=max(1,dim[0]/50)
				c1_width=max(1,dim[1]/50)
				
				face_central_lch=face_lch[int(dim[0]/2-c0_width):int(dim[0]/2+c0_width),int(dim[1]/2-c1_width):int(dim[1]/2+c1_width),:]
				#cv2.rectangle(face_roi,(dim[0]/2-dim[0]/50,dim[1]/2-dim[1]/50),(dim[0]/2-dim[0]/50+dim[0]/25,dim[1]/2-dim[1]/50+dim[1]/25),(0,255,0),2)	
				face_central_lab=face_lab[dim[0]/2-c0_width:dim[0]/2+c0_width,dim[1]/2-c1_width:dim[1]/2+c1_width,:]
				skin_L,skin_c,skin_h=np.mean(face_central_lch, axis=(0,1))
				skin_L,skin_a,skin_b=np.mean(face_central_lab, axis=(0,1))
				if len(face_central_lab)<1:
					print("empty face detection, please double check. Program exits")
					sys.exit(2)
				#cv2.namedWindow("img",cv2.WINDOW_NORMAL)
				#cv2.imshow('img',img)
				#cv2.waitKey(1000)	
				#face_pr=np.exp(-(float(np.power((avg_face_lch[0]-181),2))/(2*np.power(130,2)) + float(np.power((avg_face_lch[1]-32),2))/(2*np.power(11,2)) + float(np.power((avg_face_lch[2]-34),2))/(2*np.power(8,2))))
				#face_pr=np.exp(-(float(np.power((avg_face_lch[1]-32),2))/(2*np.power(11,2)) + float(np.power((avg_face_lch[2]-34),2))/(2*np.power(8,2))))
				img_skin_pr=np.exp(-(float(np.power((skin_c-32),2))/(2*np.power(11,2)) + float(np.power((skin_h-5),2))/(2*np.power(8,2))))
				
				face_gray= gray[y:y + h, x:x + w]
				face_gray_laplacian = cv2.Laplacian(face_gray, cv2.CV_64F)
				mu_face, std_face = cv2.meanStdDev(face_gray_laplacian)
				face_gray_sharpness = np.log2(float(std_face.item() * std_face.item()))
				face_sharpness.append(face_gray_sharpness)
				hue = face_hsv[:, :, 0]
				brightness = face_hsv[:, :, 2]
				face_gray_roi = gray[y:y + h, x:x + w]
				face_roi=img[y:y + h, x:x + w]
				#face_ratio_temp = np.log2(float(h * w) / float(img_height * img_weight))
				face_ratio_temp = float(h * w) / float(img_height * img_weight)
				#eyes = self.eye_cascade.detectMultiScale(face_gray_roi)
				eyes = self.eye_cascade.detectMultiScale(gray)
				if len(eyes):
					for (ex, ey, ew, eh) in eyes:
						eye_signal_face.append(np.average(gray[ey:ey + eh, ex:ex + ew]))
						#cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
					eye_signal.append(np.average(eye_signal_face))
					img_eye_signal = np.average(eye_signal)

				hue_mask = np.ma.masked_equal(hue, -1).compressed()
				brightness_mask = np.ma.masked_equal(brightness, -1).compressed()
				#face_skin_mask = np.ma.masked_equal(img, [-1,-1,-1]).compressed()
				face_hue.append(np.average(hue_mask))
				face_brightness.append(np.average(brightness_mask))
				face_SNR.append(20 * np.log10(np.average(brightness_mask) / np.std(brightness_mask)))
				face_ratio.append(face_ratio_temp)
				face_skin_a.append(skin_a)
				face_skin_b.append(skin_b)
				face_skin_c.append(skin_c)
				face_skin_h.append(skin_h)
				skin_pr.append(img_skin_pr)
			img_skin_pr = np.average(skin_pr)
			img_skin_a = np.average(face_skin_a)
			img_skin_b = np.average(face_skin_b)
			img_skin_c = np.average(face_skin_c)
			img_skin_h = np.average(face_skin_h)
			img_face_worstSNR = np.amin(face_SNR)
			img_face_sharpness = np.average(face_sharpness)


		self.feature_vector = [face_num, img_skin_pr, img_skin_a, img_skin_b, img_face_sharpness,img_face_worstSNR,
							   img_gray_ep,  img_sharpness]
		#print("face_SNR:"+repr(img_face_worstSNR))
		print "current image feature:", self.feature_vector
		return self.feature_vector
	
	def calEntroy(self, img, upper=240,lower=150, ROI_ratio=0.5):
		w,h=img.shape
		ROI_ratio = min(1, max(0.25, ROI_ratio))
		width_radius=max(1,w*0.5*np.sqrt(ROI_ratio))
		height_radius=max(1,h*0.5*np.sqrt(ROI_ratio))
		img_ROI = img[int(w/2-width_radius):int(w/2+width_radius), int(h/2-height_radius):int(h/2+height_radius)]
		gray_Test = np.ma.masked_outside(img_ROI, lower, upper).compressed()
		gray_hist =np.histogram(gray_Test,bins=xrange(1,256))
		gray_pd=gray_hist[0].astype(float)/len(gray_Test)
		img_gray_ep=-np.sum(np.multiply(gray_pd,np.ma.log2(gray_pd).data))
		y = gray_pd
		bins = xrange(1, len(y)+1)
		'''
		l = plt.plot(bins, y, 'r--', linewidth=1)
		plt.xlabel('graylevel')
		plt.ylabel('Probability')
		plt.title(r'image PDF ')
		plt.axis([1, 256, 0, max(y)])
		plt.grid(True)
		'''
		cur_feature_vector = self.feature_vector
		basePath = os.path.dirname(__file__)
		#print "current path of current file:", basePath
		parentPath = os.path.abspath(os.path.join(basePath,".."))
		#print "parent path of current file:",parentPath
		entropyHist_Dir = os.path.abspath(os.path.join(parentPath , 'data', 'entropy_hist'))
		#print entropyHist_Dir
		if not os.path.exists(entropyHist_Dir):
			os.mkdir(entropyHist_Dir)
		entropyHistimg_Name = "".join([entropyHist_Dir,"/",self.inputimagefile.split('/')[-1].split('.')[0]
									   + '{:-%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())+'.png'])
		#plt.savefig(entropyHistimg_Name)
		return img_gray_ep

	def enhance(self, ref_feature_vector=[2, 0.5, 19, 20, 7,4,7.05,10], enhance = True):
		bool_entropy = False
		bool_sharpen = False
		bool_denoise = False
		bool_cc = False


		cur_feature_vector = self.feature_vector
		basePath = os.path.dirname(__file__)
		#print 'Base', basePath
		parentPath = os.path.abspath(os.path.join(basePath,".."))
		enhanceimg_Dir = os.path.abspath(os.path.join(parentPath , 'data', 'enhance'))
		#print ehDir
		self.enhancedDirectory = enhanceimg_Dir

		#enhanceimg_Dir="../data/enhance/"
		if not os.path.exists(enhanceimg_Dir):
			os.mkdir(enhanceimg_Dir)

		enhancedImageFileName = "".join([enhanceimg_Dir,"/",self.inputimagefile.split('/')[-1].split('.')[0] + '{:-%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())+'.jpg'])
		img_global_entropy=cur_feature_vector[-2]
		ref_global_entropy=ref_feature_vector[-2]
		if img_global_entropy<ref_global_entropy:
			cliplimit_value = max(min(2,float(ref_global_entropy-img_global_entropy)*10),1)

			#ling
			self.enhance_constrast_brightness(cliplimit_th=cliplimit_value)
			bool_entropy = True

			#cv2.startWindowThread()
			#cv2.namedWindow("Contrasted",cv2.WINDOW_NORMAL)
			#cv2.imshow("Contrasted",self.image)

		if cur_feature_vector[0]<1:
			print("no face detected in this bad image so enhancement would focus on non-face feature such as contrast and brightness")
			#if os.path.isfile(enhanceimg_Name):
			print enhancedImageFileName, "is generated and saved"
			cv2.imwrite(enhancedImageFileName ,self.image)
			#else:
			print "Enhaced images is saved as", enhancedImageFileName
		else:
			img_face_sharpness=cur_feature_vector[4]
			ref_face_sharpness=ref_feature_vector[4]

			if img_face_sharpness<ref_face_sharpness:
				self.img_sharpen()
				bool_sharpen = True
				#cv2.startWindowThread()
				#cv2.namedWindow("Sharpened",cv2.WINDOW_NORMAL)
				#cv2.imshow("Sharpened",self.image)
				print ("current image sharpness:"+ repr(img_face_sharpness) +" < "+"reference image sharpness:" + repr(ref_face_sharpness))
				print("image sharpening is performed due to blurred image")
			#cv2.waitKey(10000)

			img_face_worstSNR=cur_feature_vector[5]
			ref_face_worstSNR=ref_feature_vector[5]
			if img_face_worstSNR<ref_face_worstSNR:
				self.skin_noise_reduction()
				bool_denoise = True
				print ("current face SNR:"+ repr(img_face_worstSNR) +" < "+"reference image SNR:" + repr(ref_face_worstSNR))
				print("image denoise is performed due to noisy face")


			img_face_color_pr=cur_feature_vector[1]
			ref_face_color_pr=ref_feature_vector[1]
			ref_face_skin_a=ref_feature_vector[2]
			ref_face_skin_b=ref_feature_vector[3]

			if img_face_color_pr<ref_face_color_pr:
				self.skin_color_correction(ref_face_skin_a, ref_face_skin_b)
				bool_cc = True
				print ("current face color accuracy:"+ repr(img_face_color_pr) +" < "+"reference face color accuracy:" + repr(ref_face_color_pr))
				print("face color correction is performed due to inaccurate face skin tone color")

			if bool_entropy == True or bool_denoise == True or bool_sharpen == True or bool_cc == True:
				#cv2.startWindowThread()
				#cv2.namedWindow("Enhanced",cv2.WINDOW_NORMAL)
				#cv2.imshow("Enhanced",self.image)
				cv2.imwrite(enhancedImageFileName ,self.image)
				if os.path.isfile(enhancedImageFileName):
					print enhancedImageFileName, "is generated and saved"
			else:
				print "No need to perform image enhancement on this image, please try another one"

		return enhancedImageFileName



	def enhance_constrast_brightness(self,gridsize = 8, cliplimit_th=5):
		faces=self.face_ROI
		img= self.image
		ori_img= copy.deepcopy(img)
		#cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
		#cv2.imshow("Original", ori_img)
		if len(faces)>0:
			for (x,y,w, h) in faces:
				face_roi = copy.deepcopy(img[y:y + h, x:x + w, :])
				#cv2.rectangle(face_roi,(x,y),(x+w,y+h),(0,255,255),2)
				#cv2.startWindowThread()
				#cv2.namedWindow("face_low_contast",cv2.WINDOW_NORMAL)
				#cv2.imshow("face_low_contast",face_roi)
			self.feature_vector[0] = len(faces)
		#cv2.waitKey(0)
		img_lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
		img_lch=  lab2lch(img_lab)
		lab_planes = cv2.split(img_lab)
		clahe = cv2.createCLAHE(clipLimit=cliplimit_th,tileGridSize=(gridsize,gridsize))
		lab_planes[0] = clahe.apply(lab_planes[0])
		lab = cv2.merge(lab_planes)
		img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
		img_enhanced2= cv2.bilateralFilter(img_enhanced,9,10, 10)
		gray_enhanced = cv2.cvtColor(img_enhanced2, cv2.COLOR_BGR2GRAY)
		faces = self.face_cascade.detectMultiScale(gray_enhanced , 1.3, 5)
		if len(faces)>0:
			for (x,y,w, h) in faces:
				face_roi_enhance = copy.deepcopy(img_enhanced2[y:y + h, x:x + w, :])
				#cv2.rectangle(face_roi_enhance,(x,y),(x+w,y+h),(0,255,255),2)
				#cv2.startWindowThread()
				#cv2.namedWindow("face_highcontrast",cv2.WINDOW_NORMAL)
				#cv2.imshow("face_highcontrast", face_roi_enhance)
			self.face_ROI=faces

		#cv2.namedWindow("contrast",cv2.WINDOW_NORMAL)
		#cv2.imshow("contrast",img_enhanced2)
		#cv2.waitKey(5000)

		#img=self.image


		#cv2.waitKey(0)
		#cv2.imwrite("../data/training/mugdha/enhance_image.jpg",bgr)


		#self.face_ROI=faces
		self.image= img_enhanced2



	def skin_noise_reduction(self):
		faces=self.face_ROI
		img=self.image
		for (x, y, w, h) in faces:
			face_roi = copy.deepcopy(img[y:y + h, x:x + w, :])

			face_lab=rgb2lab(face_roi)
			#face_lab=
			face_lch=lab2lch(face_lab)
			face_pr=np.exp(-(np.power((face_lch[:,:,1]-32),2).astype(float))/(2*np.power(11,2) + np.power((face_lch[:,:,2]-5),2).astype(float)/(2*np.power(8,2))))
			dim = face_pr.shape
			c0_width=max(1, float(dim[0])/50)
			c1_width=max(1, float(dim[1])/50)
			face_central_pr= face_pr[int(dim[0]/2-c0_width):int(dim[0]/2+c0_width),int(dim[1]/2-c1_width):int(dim[1]/2+c1_width)]
			max_denoise_color_sigma=100*np.mean(face_central_pr, axis=(0,1))
			denoise_spatial_sigma=100*np.mean(face_central_pr, axis=(0,1))
			#max_denoise_color_sigma=5
			#denoise_spatial_sigma=10
			face_smooth = cv2.bilateralFilter(face_roi,9,denoise_spatial_sigma, max_denoise_color_sigma)
			#cv2.startWindowThread()
			#cv2.namedWindow("face_smooth",cv2.WINDOW_NORMAL)
			#cv2.imshow("face_smooth", face_smooth)

			img[y:y + h, x:x + w, :] = face_smooth

		self.image=img


	def face_sharpen(self):
		faces=self.face_ROI
		img=self.image
		for (x, y, w, h) in faces:
			face_roi = copy.deepcopy(img[y:y + h, x:x + w, :])
			face_yuv=cv2.cvtColor(face_roi, cv2.COLOR_RGB2YUV)
			#face_yuv=rgb2yuv(face_roi)
			#face_roi2=yuv2rgb(face_yuv)*256
			face_roi2=cv2.cvtColor(face_yuv, cv2.COLOR_YUV2RGB)
			#cv2.namedWindow("roi1",cv2.WINDOW_NORMAL)
			#cv2.imshow("roi1", face_roi2)
			#cv2.startWindowThread()
			#cv2.namedWindow("unsharpen",cv2.WINDOW_NORMAL)
			#cv2.imshow("unsharpen", face_roi)



			kernel = np.zeros( (9,9), np.float32)
			kernel[4,4] = 2.0   #Identity, times two!

			#Create a box filter:
			boxFilter = np.ones( (9,9), np.float32) / 81.0

			#Subtract the two:
			kernel = kernel - boxFilter

			custom = cv2.filter2D(face_yuv[:,:,0], -1, kernel)

			face_yuv[:,:,0]=custom
			#face_roi_sharpen=(yuv2rgb(face_yuv))
			face_roi_sharpen=cv2.cvtColor(face_yuv, cv2.COLOR_YUV2RGB)
			#cv2.namedWindow("Sharpen",cv2.WINDOW_NORMAL)
			#cv2.imshow("Sharpen", face_roi_sharpen)
			#cv2.waitKey(10000)
			img[y:y + h, x:x + w, :] = face_roi_sharpen
		self.image=img

	def img_sharpen(self):
		img=self.image
		img_yuv=cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		kernel = np.zeros( (9,9), np.float32)
		kernel[4,4] = 2.0   #Identity, times two!
		#Create a box filter:
		boxFilter = np.ones( (9,9), np.float32) / 81.0
		#Subtract the two:
		kernel = kernel - boxFilter
		custom = cv2.filter2D(img_yuv[:,:,0], -1, kernel)
		img_yuv[:,:,0]=custom
		#face_roi_sharpen=(yuv2rgb(face_yuv))
		img_sharpen=cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
		#cv2.namedWindow("Sharpen",cv2.WINDOW_NORMAL)
		#cv2.imshow("Sharpen", img_sharpen)
		#cv2.waitKey(10000)
		#img[y:y + h, x:x + w, :] = face_roi_sharpen
		self.image=img_sharpen

	def skin_color_correction(self, ref_a, ref_b):
		faces=self.face_ROI
		img=self.image
		for (x, y, w, h) in faces:
			face_roi = copy.deepcopy(img[y:y + h, x:x + w, :])
			#cv2.startWindowThread()
			#cv2.namedWindow("face_wrong_color", cv2.WINDOW_NORMAL)
			#cv2.imshow("face_wrong_color", face_roi)
			#face_lab=cv2.cvtColor(face_roi, cv2.COLOR_RGB2LAB)
			face_lab=rgb2lab(face_roi)
			face_lch=lab2lch(face_lab)
			face_pr=np.exp(-(np.power((face_lch[:,:,1]-32),2).astype(float))/(2*np.power(11,2) + np.power((face_lch[:,:,2]-5),2).astype(float)/(2*np.power(8,2))))
			a_diff = np.multiply((face_lab[:,:,1]-ref_a), face_pr*0.33)
			b_diff = np.multiply((face_lab[:,:,2]-ref_b), face_pr*0.33)
			face_lab[:,:,1] = face_lab[:,:,1] + a_diff
			face_lab[:,:,2] = face_lab[:,:,2] + b_diff
			face_skin_corrected = lab2rgb(face_lab)
			face_skin_corrected_new = (face_skin_corrected *256).astype(int).clip(0,255)
			#face_skin_corrected =cv2.cvtColor(face_lab, cv2.COLOR_lab2rgb)
			#Ling
			#cv2.namedWindow("Face_color_corrected",cv2.WINDOW_NORMAL)
			#cv2.imshow("Face_color_corrected", face_skin_corrected)

			#cv2.waitKey(30000)
			#Ling

			img[y:y + h, x:x + w, :] = face_skin_corrected_new
			#cv2.namedWindow("Whole_Image",cv2.WINDOW_NORMAL)
			#cv2.imshow("Whole_Image", img)
			#cv2.waitKey(10000)

		self.image=img


			
			
		
		



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                               Main Method                                    """
"""                                                                              """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
	#testimage = '../data/training/test/original.jpg'
	#Ling
	testimage = os.path.abspath('../data/training/0/m.JPG')
	face_hcxml = './xml/HAAR_FACE.xml'
	#eye_hcxml = './xml//HAAR_EYE.xml'
	eye_hcxml = './xml/haarcascade_eye_tree_eyeglasses.xml'
	im_extractor = ImageFeatureExtractor(testimage, face_hcxml, eye_hcxml)
	# im_extractor.initialize()
	img_feature=im_extractor.extract()
	good_feature_vector=[2, 0.5, 19, 20, 7,8 ,7.05,10]
	enhanced = im_extractor.enhance(ref_feature_vector=good_feature_vector)
	from PIL import Image
	imOld = Image.open(testimage).rotate(270)
	imOld.show()
	imNew =Image.open(enhanced)
	imNew.show()
	'''
	im_extractor.enhance_constrast_brightness()
	im_extractor.face_sharpen()
	im_extractor.skin_noise_reduction()
	im_extractor.skin_color_correction(19, 30)
	#im_extractor.skin_color_correction(19, 20)
	#print im_extractor.feature_vector
	'''