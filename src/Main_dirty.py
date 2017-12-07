import matplotlib#needed to make sure TKinter doesn't crash!
matplotlib.use('TkAgg') #needed to make sure TKinter doesn't crash!
from sklearn.metrics import confusion_matrix
from Trainer.KNNTrainer import KNNTrainer
from Trainer.RandomForestTrainer import RandomForestTrainer
from Trainer.SVMTrainer import SVMTrainer
from Trainer.TrainingFileReader import TrainingFileReader
from Validators.CrossValidator import CrossValidator
from Tkinter import Tk
from tkFileDialog import askopenfilename, askdirectory
from ImageFeatureExtractor.ImageFeatureExtractor import ImageFeatureExtractor
import numpy as np
<<<<<<< HEAD
import os

=======
import sys
import os
from Evaluator import Evaluator
>>>>>>> 8398dc53462cf58b443693335f6ed72c0698f8b2

class Main:

	def __init__(self):
		imageVectorList = []
		imageLabelList = []
		imageRefVectorList=[]
		'''
		classifiers = [KNNTrainer(9),
		               RandomForestTrainer(10),
		               SVMTrainer(None)]
		'''			   
		classifiers = [SVMTrainer(None)]
		trainingDirectory = Main.pickDirectory('Select Training directory')
		testingDirectory = Main.pickDirectory('Select Test directory')

<<<<<<< HEAD
		# directory = raw_input("Please enter training directory containing labelled images: ")
		# directory = "./data/training/ling"
		testingDirectory = Main.pickTestDirectory()
		trainingDirectory = Main.pickTrainingDirectory()
		print trainingDirectory
		print testingDirectory
		imageVectorList, imageLabelList, imageTrainFileList= TrainingFileReader.extractTrainingData(trainingDirectory)
		imageTestList, imageTestLabelList, imageTestFileList= TrainingFileReader.extractTrainingData(testingDirectory)
		#imageVectorList_np=np.asarray(imageVectorList)
		imageVectorList_short=np.asarray(imageVectorList)[:,[0,1,4,5,6,7]].tolist()
		#imageTestList_np=np.asarray(imageVectorList)
		imageTestList_short=np.asarray(imageVectorList)[:,[0,1,4,5,6,7]].tolist()
		for classifier in classifiers:
			classifier.buildModel(imageVectorList_short, imageLabelList)
			predictions = CrossValidator(classifier, imageVectorList_short, imageLabelList).getPredictions()
			tn, fp, fn, tp = confusion_matrix(imageLabelList, predictions).ravel()
			t_predictions= classifier.model.predict(imageTestList_short)
=======
		print 'Reading training images...'
		imageVectorList, imageLabelList, imgTrainFileList= TrainingFileReader.extractTrainingData(trainingDirectory)
		print 'Reading test images...'
		imageTestList, imageTestLabelList, imgTestFileList= TrainingFileReader.extractTrainingData(testingDirectory)

		t_predictions = []
		for classifier in classifiers:
			print 'Building Model'
			classifier.buildModel(imageVectorList, imageLabelList)
			print 'Cross Validating'
			predictions = CrossValidator(classifier, imageVectorList, imageLabelList).getPredictions()
			tn, fp, fn, tp = confusion_matrix(imageLabelList, predictions).ravel()
			print 'Testing'
			t_predictions= classifier.model.predict(imageTestList)
>>>>>>> 8398dc53462cf58b443693335f6ed72c0698f8b2
			tn1, fp1, fn1, tp1 = confusion_matrix(imageTestLabelList, t_predictions).ravel()
			print ("Confusion Matrix for Cross Validation %s" %classifier.name)
			print ("\tGood\tBad")
			print("Good %i\t\t%i" %(tp, fn))
			print("Bad  %i\t\t%i" % (fp, tn))
<<<<<<< HEAD
			print ("Classification Accuracy for Training:%f" %(float(tp+tn)/(tp+fn+fp+tn)))
=======
			print ("Classification Accuracy for training:%.4f" %(float(tp+tn)/(tp+fn+fp+tn)))
>>>>>>> 8398dc53462cf58b443693335f6ed72c0698f8b2
			CrossValidator(classifier, imageVectorList, imageLabelList).printAccuracy()
			
			print ("Confusion Matrix for Test Image for %s" %classifier.name)
			print ("\tGood\tBad")
			print("Good %i\t\t%i" %(tp1, fn1))
			print("Bad  %i\t\t%i" % (fp1, tn1))
			print ("Classification Accuracy for Testing:%.4f" %(float(tp1+tn1)/(tp1+fn1+fp1+tn1)))
		
		for i in range(len(imageVectorList)):
			if predictions[i] == "1":
				imageRefVectorList.append(imageVectorList[i])
		
		ref_feature_vector = np.mean(np.asarray(imageRefVectorList)	, axis=0)	
		
		if len(ref_feature_vector)!=8:
			print("invalid feature vector")
			sys.exit(2)
<<<<<<< HEAD
		
		
		if len(t_predictions) != len(imageTestFileList):
			print("ummatch predictions and test images number")
			sys.exit(2)
			
		for i in range(len(t_predictions)):
			if t_predictions[i] == "0":
				imgbad=ImageFeatureExtractor(imageTestFileList[i])
				imgbad.initialize()
				imgbad.extract()
				#good_feature_vector=[2, 0.5, 19, 20, 7,8 ,7.05,10]
				imgbad.enhance(ref_feature_vector)
			else:
				basePath = os.path.dirname(__file__)
				train_good_path=os.path.abspath(os.path.join(basePath,"data","training","1"))	
			
	@staticmethod
	def pickTestDirectory():

		dirPickTitle = "Select test data directory"
		# instantiate a Tk window
		root = Tk()

			# set the title of the window
		root.title(dirPickTitle)
		root.update()

			# Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
			# show an "Open" dialog box and return the path to the selected file
		directory = askdirectory(initialdir = "/",
			                    title = dirPickTitle)
		root.destroy()
		return directory
	
		
=======

		if len(t_predictions) != len(imgTestFileList):
			print("ummatch predictions and test images number")
			sys.exit(2)

		agentFileList = []
		agentLabels = []
		enhanceDir = trainingDirectory
		for i in range(len(t_predictions)):
			agentFile = imgTestFileList[i]
			predictedLabel  = '1'
			if t_predictions[i] == "0":
				imgbad= ImageFeatureExtractor(imgTestFileList[i])
				# imgbad.initialize()
				imgbad.extract()
				#good_feature_vector=[2, 0.5, 19, 20, 7,8 ,7.05,10]
				agentFile = imgbad.enhance(ref_feature_vector)
				enhanceDir = os.path.dirname(agentFile)
				predictedLabel = '0'
			else:
				basePath = os.path.dirname(__file__)
				train_good_path=os.path.abspath(os.path.join(basePath,"data","training","1"))
			agentFileList.append(agentFile)
			agentLabels.append(predictedLabel)
		print 'Training', trainingDirectory
		evaluator = Evaluator(trainingDirectory, enhanceDir)
		userLabels = evaluator.getUserEvaluation(agentFileList)
		print Evaluator.calculateScore(agentLabels,userLabels)
>>>>>>> 8398dc53462cf58b443693335f6ed72c0698f8b2

	# @staticmethod
	# def crossValidate(classifier, imageVectorList, imageLabelList):
	# 	#Cross-Validate
	# 	predictions = CrossValidator(classifier, imageVectorList, imageLabelList).getPredictions()
	# 	return predictions
	#
	# @staticmethod
	# def test(classifier, imageVectorList, imageLabelList):
	# 	pass
	#
	# @staticmethod
	# def printPredictions(title,labelList, predictions):
	# 	pass
	@staticmethod
	def pickDirectory(title=''):

		dirPickTitle = title
		# instantiate a Tk window
		root = Tk()

			# set the title of the window
		root.title(dirPickTitle)
		root.update()

		# Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
		# show an "Open" dialog box and return the path to the selected file
		directory = askdirectory(initialdir = "/",
			                    title = dirPickTitle)
		root.destroy()
		return directory



if __name__ == "__main__":
	Main()
