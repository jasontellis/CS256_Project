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
from ImageEnhancer import ImageEnhancer
import numpy as np
import sys
import os
from Evaluator import Evaluator

class Main:

	def __init__(self):
		imageVectorList = []
		imageLabelList = []
		imageRefVectorList=[]

		classifiers = [KNNTrainer(5),
		               SVMTrainer(None),
		               RandomForestTrainer(10),
		               ]
		testingDirectory = []
		print 'Select training directory...'
		trainingDirectory = Main.pickDirectory('Select Training directory')
		print 'Reading training images...'
		imageVectorList, imageLabelList, imgTrainFileList= TrainingFileReader.extractTrainingData(trainingDirectory)


		isContinue = 'y'
		while isContinue.lower() == 'y':

			print 'Select directory with images to process:'
			testingDirectory = Main.pickDirectory('Select directory with images to process')
			print 'Reading images to process...'
			imageTestList, imageTestLabelList, imgTestFileList = TrainingFileReader.extractTrainingData(
				testingDirectory)
			t_predictions = []
			for classifier in classifiers:
				print 'Building Model'
				classifier.buildModel(imageVectorList, imageLabelList)
				print 'Cross Validating'
				predictions = CrossValidator(classifier, imageVectorList, imageLabelList).getPredictions()
				tn, fp, fn, tp = confusion_matrix(imageLabelList, predictions).ravel()
				print 'Testing'
				t_predictions= classifier.model.predict(imageTestList)
				tn1, fp1, fn1, tp1 = confusion_matrix(imageTestLabelList, t_predictions).ravel()
				print ("Confusion Matrix for Cross Validation %s" %classifier.name)
				print ("\tGood\tBad")
				print("Good %i\t\t%i" %(tp, fn))
				print("Bad  %i\t\t%i" % (fp, tn))
				# print ("Classification Accuracy for training:%.4f" %(float(tp+tn)/(tp+fn+fp+tn)))
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
			print 'Reference Feature Vector:',ref_feature_vector

			if len(ref_feature_vector)!=8:
				print("Invalid feature vector")
				sys.exit(2)

			if len(t_predictions) != len(imgTestFileList):
				print("Mismatch between count of predictions and test images")
				sys.exit(2)

			outputFileList = []
			agentLabels = []
			enhancedImageDirectory = trainingDirectory
			countOfClassifiedBad =  0
			for i in range(len(t_predictions)):
				outputFile = imgTestFileList[i]
				predictedLabel  = t_predictions[i]
				if predictedLabel == "0":
					countOfClassifiedBad += 1
					imgbad = ImageEnhancer(imgTestFileList[i])
					#good_feature_vector=[2, 0.5, 19, 20, 7,8 ,7.05,10]
					outputFile = imgbad.enhance(ref_feature_vector)
					enhancedImageDirectory = os.path.dirname(outputFile)
				else:
					basePath = os.path.dirname(__file__)
					train_good_path = os.path.abspath(os.path.join(basePath,"data","training","1"))
				outputFileList.append(outputFile)
				agentLabels.append(predictedLabel)
			evaluator = Evaluator(trainingDirectory, enhancedImageDirectory)
			print 'Agent Statistics: \nTotal Images:%i \nClassified Good:%i \nClassified Bad:%i' % (
			len(t_predictions), len(t_predictions) - countOfClassifiedBad, countOfClassifiedBad)
			print 'Evaluate processed images'
			userLabels = evaluator.getUserEvaluation(outputFileList)

			print 'Score: %.4f'%Evaluator.calculateScore(agentLabels,userLabels)
			isContinue = raw_input("Do you wish to continue? y: Yes, any other key to exit: ")


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
