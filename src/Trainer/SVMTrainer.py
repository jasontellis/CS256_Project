from sklearn.svm import SVC

from AbstractTrainer import AbstractTrainer


class SVMTrainer(AbstractTrainer):
	def __init__(self, parameter):
		"""

		:param parameter:
		"""
		AbstractTrainer.__init__(self, parameter)
		self.name = "Support Vector Machine Classifier"
		self.model = SVC(kernel = 'linear')
