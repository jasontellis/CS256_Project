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
	'''
	Extracts feature Vector from given Image
	'''

	def __init__(self, imageFileName, facecascxml_path="", eyecascxml_path=""):
		face_hcxml = './xml/HAAR_FACE.xml'
		# eye_hcxml = './xml//HAAR_EYE.xml'
		eye_hcxml = './xml/haarcascade_eye_tree_eyeglasses.xml'
		self.EYE_XML = os.path.abspath(os.path.join(os.path.dirname(__file__), 'xml', 'haarcascade_eye_tree_eyeglasses.xml'))
		self.FACE_XML = os.path.abspath(os.path.join(os.path.dirname(__file__), 'xml', 'HAAR_FACE.xml'))
		self.feature_vector = []
		self.face_ROI=[]
		self.image = None
		self.inputimagefile = imageFileName
		self.inputfacecascxml = os.path.abspath(facecascxml_path)
		self.inputeyecasxml = os.path.abspath(eyecascxml_path)

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
					# print("Empty face detection, please double check. Program exits")
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
		# print "current image feature:", self.feature_vector
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
	#enhanced = im_extractor.enhance(ref_feature_vector=good_feature_vector)
	from PIL import Image
	imOld = Image.open(testimage).rotate(270)
	imOld.show()
	#imNew =Image.open(enhanced)
	#imNew.show()
	'''
	im_extractor.enhance_constrast_brightness()
	im_extractor.face_sharpen()
	im_extractor.skin_noise_reduction()
	im_extractor.skin_color_correction(19, 30)
	#im_extractor.skin_color_correction(19, 20)
	#print im_extractor.feature_vector
	'''