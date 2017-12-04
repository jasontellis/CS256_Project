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


class Main:

	def __init__(self):
		imageVectorList = []
		imageLabelList = []

		classifiers = [KNNTrainer(9),
		               RandomForestTrainer(10),
		               SVMTrainer(None)]

		# directory = raw_input("Please enter training directory containing labelled images: ")
		# directory = "./data/training/ling"

		trainingDirectory = Main.pickTrainingDirectory()
		print trainingDirectory
		imageVectorList, imageLabelList = TrainingFileReader.extractTrainingData(trainingDirectory)

		for classifier in classifiers:
			classifier.buildModel(imageVectorList, imageLabelList)
			predictions = CrossValidator(classifier, imageVectorList, imageLabelList).getPredictions()
			tn, fp, fn, tp = confusion_matrix(imageLabelList, predictions).ravel()
			print ("Confusion Matrix for %s" %classifier.name)
			print ("\tGood\tBad")
			print("Good %i\t\t%i" %(tp, fn))
			print("Bad  %i\t\t%i" % (fp, tn))
			CrossValidator(classifier, imageVectorList, imageLabelList).printAccuracy()

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
		return directory


if __name__ == "__main__":
	Main()
