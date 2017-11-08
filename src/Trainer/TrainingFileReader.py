import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import Constants.Constants as Constants


class TrainingFileReader:
	@staticmethod
	def getFileList(directory):
		"""
		Returns list of images in a training directory with associated label

		:param directory: File Path
		:return: Returns a list of image filenames keyed by label in a directory
		"""
		labelledFileList = []
		for label in Constants.CLASS_LABELS:

			labelledDirectory = os.path.join(directory, label)
			if not os.path.isdir(labelledDirectory):
				raise ValueError("No directory found with label %s" % label)
			for file in os.listdir(labelledDirectory):
				fileName = os.path.join(labelledDirectory, file)
				if os.path.isfile(fileName):
					labelledFileList.append((fileName, label))

		return labelledFileList


if __name__ == '__main__':
	print TrainingFileReader.getFileList('../data/training/ling/')
