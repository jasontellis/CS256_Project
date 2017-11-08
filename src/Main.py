from ImageFeatureExtractor import ImageFeatureExtractor
from Trainer.KNNTrainer import KNNTrainer
from Trainer.RandomForestTrainer import RandomForestTrainer
from Trainer.SVMTrainer import SVMTrainer
from Trainer.TrainingFileReader import TrainingFileReader
from Validators.CrossValidator import CrossValidator


class Main:
	def __init__(self):
		imageVectorList = []
		imageLabelList = []

		classifiers = [KNNTrainer(5),
		               RandomForestTrainer(10),
		               SVMTrainer(None)]

		directory = raw_input("Please enter training directory containing labelled images: ")
		trainingFiles = TrainingFileReader.getFileList(directory)
		for (trainingFile, label) in trainingFiles:
			imageVector = ImageFeatureExtractor(trainingFile).getFeature()
			imageVectorList.append(imageVector)
			imageLabelList.append(label)

		for classifier in classifiers:
			classifier.buildModel(imageVectorList, imageLabelList)
			CrossValidator(classifier, imageVectorList, imageLabelList).printAccuracy()


if __name__ == "__main__":
	Main()
