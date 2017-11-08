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

		classifiers = [KNNTrainer(9),
		               RandomForestTrainer(10),
		               SVMTrainer(None)]

		# directory = raw_input("Please enter training directory containing labelled images: ")
		directory = "./data/training/ling"
		trainingFiles = TrainingFileReader.getFileList(directory)
		for (trainingFile, label) in trainingFiles:
			print trainingFile
			imageVector = ImageFeatureExtractor(trainingFile,
			                                    './ImageFeatureExtractor/xml/haarcascade_frontalface_default.xml',
			                                    './ImageFeatureExtractor/xml/haarcascade_eye.xml').extract()
			imageVectorList.append(imageVector)
			imageLabelList.append(label)
		print imageVectorList
		print imageLabelList

		for classifier in classifiers:
			classifier.buildModel(imageVectorList, imageLabelList)
			CrossValidator(classifier, imageVectorList, imageLabelList).printAccuracy()


if __name__ == "__main__":
	Main()
