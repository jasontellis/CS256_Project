from sklearn.metrics import confusion_matrix

from Trainer.KNNTrainer import KNNTrainer
from Trainer.RandomForestTrainer import RandomForestTrainer
from Trainer.SVMTrainer import SVMTrainer
from Trainer.TrainingFileReader import TrainingFileReader
from Validators.CrossValidator import CrossValidator


class Main:
	def __init__(self):
		imageVectorList = []
		imageLabelList = []

		classifiers = [KNNTrainer(9),
		               RandomForestTrainer(10),
		               SVMTrainer(None)]

		# directory = raw_input("Please enter training directory containing labelled images: ")
		directory = "./data/training/ling"

		imageVectorList, imageLabelList = TrainingFileReader.extractTrainingData(directory)

		for classifier in classifiers:
			classifier.buildModel(imageVectorList, imageLabelList)
			predictions = CrossValidator(classifier, imageVectorList, imageLabelList).getPredictions()
			tn, fp, fn, tp = confusion_matrix(imageLabelList, predictions).ravel()
			print ("Confusion Matrix for %s" %classifier.name)
			print ("\tGood\tBad")
			print("Good %i\t\t%i" %(tp, fn))
			print("Bad  %i\t\t%i" % (fp, tn))
			CrossValidator(classifier, imageVectorList, imageLabelList).printAccuracy()


if __name__ == "__main__":
	Main()
