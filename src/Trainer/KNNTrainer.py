from sklearn.neighbors import KNeighborsClassifier

from AbstractTrainer import AbstractTrainer


class KNNTrainer(AbstractTrainer):
	'''
	K-Nearest Neighbor based Image Classifier
	'''
	def __init__(self, parameter = 5):
		"""

		:param parameter: No. of nearest neighbor to be considered
		"""
		AbstractTrainer.__init__(self, parameter)
		self.name = "KNN Classifier"
		self.model = KNeighborsClassifier(parameter)