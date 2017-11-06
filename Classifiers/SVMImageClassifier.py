from sklearn import datasets, svm, metrics # Import datasets, classifiers and performance metrics

from AbstractImageClassifier import AbstractImageClassifier
from sklearn.svm import SVC

class SVMImageClassifier(AbstractImageClassifier):
	"""

	"""

	def __init__(self, model):
		self.model = model

	def classify(self, imageVector):
		"""
		Classifies given image as good or bad
		:param imageVector:
		:return:
		"""

