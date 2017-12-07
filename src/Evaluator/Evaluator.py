import sys
import os,os.path
rootPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'+os.sep))
sys.path.append(rootPath) #Used for importing local packages
from Utility import  Utility
import Tkinter as tk
from Tkinter import *
from PIL import Image,ImageTk
import shutil
import datetime, time
import itertools
import functools


class Evaluator:

	def __init__(self, trainingDirectory=os.path.abspath('..'+os.sep+'data'+os.sep+'training'+os.sep), evaluationDirectory=os.path.abspath('..'+os.sep+'data'+os.sep+'enhance')):
		self.evaluationDirectory = evaluationDirectory
		self.trainingDirectory = trainingDirectory
		self.userLabels = []

	def __getFilesToBeDisplayed__(self):
		'''

		:return: List of images in evaluation directory
		'''

		fileList = []
		for f in os.listdir(self.evaluationDirectory):
			if f.lower().endswith('jpg') or f.lower().endswith('jpeg') or f.lower().endswith('png'):
				fileList.append(os.path.join(self.evaluationDirectory, f))

		return fileList


	def getUserEvaluation(self, filesToBeEvaluated):
		'''

		:return: Show images in training directory
		'''

		# imageFileList = self.__getFilesToBeDisplayed__()
		self.userLabels = []
		for imageFilename in filesToBeEvaluated:
			root = Tk()
			resizedImage = Image.open(imageFilename)
			resizedImage = Evaluator.image_transpose_exif(resizedImage)
			resizedImage = resizedImage.resize((500, 500), Image.ANTIALIAS)  # The (250, 250) is (height, width)
			tkinterImg = ImageTk.PhotoImage(resizedImage)
			panel = tk.Label(root, image = tkinterImg)
			panel.pack(side = "top")
			b1 = Button(root, text = "Like", command = lambda: likeDislikeCallback("1", imageFilename, root, self))
			b1.pack()
			b2 = Button(root, text = 'Dislike', command = lambda: likeDislikeCallback("0", imageFilename, root, self))
			b2.pack()
			root.mainloop()
		return self.userLabels
	
	@staticmethod
	def image_transpose_exif(im):
		exif_orientation_tag = 0x0112 # contains an integer, 1 through 8
		exif_transpose_sequences = [  # corresponding to the following
			[],
			[Image.FLIP_LEFT_RIGHT],
			[Image.ROTATE_180],
			[Image.FLIP_TOP_BOTTOM],
			[Image.FLIP_LEFT_RIGHT, Image.ROTATE_90],
			[Image.ROTATE_270],
			[Image.FLIP_TOP_BOTTOM, Image.ROTATE_90],
			[Image.ROTATE_90],
		]

		try:
			seq = exif_transpose_sequences[im._getexif()[exif_orientation_tag] - 1]
		except Exception:
			return im
		else:
			return functools.reduce(lambda im, op: im.transpose(op), seq, im)
	
	
	@staticmethod
	def calculateScore(agentLabelList, userLabelList):
		score = 0.0
		counter = 0
		tn, tp, fn, fp = 0,0,0,0,
		if len(agentLabelList) != len(userLabelList):
			print "Count mismatch for classifier and agent labels"
			sys.exit(2)
		if len(agentLabelList) == 0 or  len(userLabelList) == 0:
			print "Zero labels"
			sys.exit(2)
		for agentLabel, userLabel in itertools.izip(agentLabelList, userLabelList):
			currScore = 0
			userLabel = str(userLabel)
			counter += 1
			if agentLabel == '0' and userLabel == '0':#Penalize a agent classified bad image that is still bad after enhancement
				currScore = -1
				tn += 1
			elif agentLabel == '0' and userLabel == '1':#Reinforce a bad image that is good after enhancement
				currScore = 2
				fp += 1
			elif agentLabel == '1' and userLabel == '0':#Penalize a classifier tagged 'good' image that is labelled bad after enhancement
				currScore = -2
				fn += 1
			elif agentLabel == '1' and userLabel == '1':#Reinforce a good image that was not anhanced and is still considered good by user
				currScore = 0
				fp += 1
			score += currScore
		score /= (2*counter) #Normalize Score
		print 'Confusion Matrix for User Evaluation vs Classifier Evaluation'
		print ("\t\t\t\tPredicted Good\tPredicted Bad")
		print("User-labelled Good \t%i\t\t%i" % (tp, fn ))
		print("User-labelled Bad  \t%i\t\t%i" % (fp, tn))
		print ("Accuracy of Agent :%.4f" % (float(tp + tn) / (tp + fn + fp + tn)))
		return score


	@staticmethod
	def resizeImage(imageFile):
		resizedImage = Image.open(imageFile)
		resizedImage = resizedImage.resize((500, 500), Image.ANTIALIAS)
		return resizedImage

	def moveImageToTraining(self, imageFile):
		fileName = os.path.basename(imageFile)
		destinationFilename = Evaluator.getTimestamp()+fileName
		destination = os.path.join(self.trainingDirectory, '1',destinationFilename)
		# print 'Moved %s to %s'%(imageFile, destination)
		shutil.copy(imageFile,destination)

	@staticmethod
	def getTimestamp():
		timestamp = datetime.datetime.fromtimestamp(time.time()  ).strftime('%Y%m%d%H%M%S')
		return timestamp


def likeDislikeCallback(choice, imageFile, tkRoot, evaluator):
	LIKE = '1'
	DISLIKE = '0'
	tkRoot.destroy()
	if choice == LIKE:
		evaluator.moveImageToTraining(imageFile)
	evaluator.userLabels.append(choice)

if __name__ == '__main__':
	print Evaluator().calculateScore(['0','0','1','1'],['0','0','1','1'])