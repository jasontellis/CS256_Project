from sklearn.neighbors import KNeighborsClassifier

from AbstractTrainer import AbstractTrainer


class KNNTrainer(AbstractTrainer):
	def __init__(self, parameter = 5):
		"""

		:param parameter: No. of nearest neighbor to be considered
		"""
		AbstractTrainer.__init__(self, parameter)
		self.name = "KNN Classifier"
		self.model = KNeighborsClassifier(parameter)


if __name__ == "__main__":
	imageVectorList = [[1, 1],
	                   [2, 2],
	                   [3, 3],
	                   [4, 4]]
	imageLabelList = [1, 0, 1, 0]
	KNNTrainer = KNNTrainer().buildModel(imageVectorList, imageLabelList)
