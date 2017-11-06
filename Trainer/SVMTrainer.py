from sklearn.svm import SVC
from AbstractTrainer import AbstractTrainer
class SVMTrainer(AbstractTrainer):

	def __init__(self):
		self.svmModel = None

	def getModel(self):
		self.svmModel = SVC()
		self.svmModel.fit()
		classLabel = self.svmModel.predict(imageVector)