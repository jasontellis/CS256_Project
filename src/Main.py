#import matplotlib#needed to make sure TKinter doesn't crash!
#matplotlib.use('TkAgg') #needed to make sure TKinter doesn't crash!
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

		# directory = raw_input("Please enter training directory containing labelled images: ")
		# directory = "./data/training/ling"
		testingDirectory = Main.pickTestDirectory()
		trainingDirectory = Main.pickTrainingDirectory()
		print trainingDirectory
		print testingDirectory
		imageVectorList, imageLabelList, imgTrainFileList= TrainingFileReader.extractTrainingData(trainingDirectory)
		imageTestList, imageTestLabelList, imgTestFileList= TrainingFileReader.extractTrainingData(trainingDirectory)
		#imageVectorList_np=np.asarray(imageVectorList)
		#imgVectorListList=np.asarray(imageVectorList)[:,[0,1,4,5,6,7]].tolist()
		#imageTestList_np=np.asarray(imageVectorList)
		#imgTestListList=np.asarray(imageVectorList)[:,[0,1,4,5,6,7]].tolist()
		for classifier in classifiers:
			classifier.buildModel(imageVectorList, imageLabelList)
			predictions = CrossValidator(classifier, imageVectorList, imageLabelList).getPredictions()
			tn, fp, fn, tp = confusion_matrix(imageLabelList, predictions).ravel()
			t_predictions= classifier.model.predict(imageTestList)
			tn1, fp1, fn1, tp1 = confusion_matrix(imageTestLabelList, t_predictions).ravel()
			print ("Confusion Matrix for %s" %classifier.name)
			print ("\tGood\tBad")
			print("Good %i\t\t%i" %(tp, fn))
			print("Bad  %i\t\t%i" % (fp, tn))
			print ("Classification Accuracy for training:%f" %(float(tp+tn)/(tp+fn+fp+tn)))
			CrossValidator(classifier, imageVectorList, imageLabelList).printAccuracy()
			
			print ("Confusion Matrix for Test Image for %s" %classifier.name)
			print ("\tGood\tBad")
			print("Good %i\t\t%i" %(tp1, fn1))
			print("Bad  %i\t\t%i" % (fp1, tn1))
			print ("Classification Accuracy for Testing:%f" %(float(tp1+tn1)/(tp1+fn1+fp1+tn1)))
		
		for i in range(len(imageVectorList)):
			if predictions[i] == "1":
				imageRefVectorList.append(imageVectorList[i])
		
		ref_feature_vector = np.mean(np.asarray(imageRefVectorList)	, axis=0)	
		
		if len(ref_feature_vector)!=8:
			print("invalid feature vector")
			sys.exit(2)
		
		
		if len(t_pedictions) != len(imgTestFileList):
			print("ummatch predictions and test images number")
			sys.exit(2)
			
		for i in range(len(t_pedictions)):
			if t_predictions[i] == "0":
				imgbad=ImageFeatureExtractor(imgTestFileList[i])
				im_extractor.initialize()
				im_extractor.extract()
				#good_feature_vector=[2, 0.5, 19, 20, 7,8 ,7.05,10]
				im_extractor.enhance(ref_feature_vector)
			else:
				basePath = os.path.dirname(__file__)
				train_good_path=os.path.abspath(os.path.join(basePath,"data","training","1"))
			
		Ling		
			
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
	
		

	@staticmethod
	def pickTrainingDirectory():

		dirPickTitle = "Select training directory"
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

	@staticmethod
	def pickImageToEnhance():
		filePickTitle = "Pick image for enhancement"

if __name__ == "__main__":
	Main()
