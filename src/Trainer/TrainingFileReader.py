import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import Constants.Constants as Constants
from ImageFeatureExtractor import ImageFeatureExtractor


class TrainingFileReader:

	@staticmethod
	def __getFileList__(directory, all=True, selectClass="1"):
		"""
		Returns list of images in a training directory with associated label

		:param directory: File Path
		:return: Returns a list of image filenames keyed by label in a directory
		"""
		labelledFileList = []
		if all==True:
			for label in Constants.CLASS_LABELS:
				directory = os.path.abspath(directory)
				labelledDirectory = os.path.join(directory, label)
				if not os.path.isdir(labelledDirectory):
					raise ValueError("No directory found with label %s" % label)
				for file in os.listdir(labelledDirectory):
					fileName = os.path.join(labelledDirectory, file)
					if os.path.isfile(fileName) and (fileName.lower().endswith('jpg') or fileName.lower().endswith('jpeg')):
						labelledFileList.append((fileName, label))
		else:				
			label = selectClass
			labelledDirectory = os.path.join(directory, label)
			if not os.path.isdir(labelledDirectory):
				raise ValueError("No directory found with label %s" % label)
			for file in os.listdir(labelledDirectory):
				fileName = os.path.join(labelledDirectory, file)
				if os.path.isfile(fileName) and (fileName.lower().endswith('jpg') or fileName.lower().endswith('jpeg')):
					labelledFileList.append((fileName, label))
		return labelledFileList

	@staticmethod
	def extractTrainingData(directory, boolAll = True, select = 1):
		"""
		Returns a list of image vectors and list of associated labels
		:return:
		"""
		imageVectorList = []
		imageLabelList = []
		imageFileList =[]
		trainingFiles = TrainingFileReader.__getFileList__(directory,boolAll,select)
		for (trainingFile, label) in trainingFiles:
			imageVector = ImageFeatureExtractor(trainingFile,
			                                    './ImageFeatureExtractor/xml/HAAR_FACE.xml',
			                                    './ImageFeatureExtractor/xml/HAAR_EYE.xml').extract()
			imageVectorList.append(imageVector)
			imageLabelList.append(label)
			imageFileList.append(trainingFile)
		return imageVectorList, imageLabelList, imageFileList


if __name__ == '__main__':
	print TrainingFileReader.__getFileList__('../data/training/ling/')
