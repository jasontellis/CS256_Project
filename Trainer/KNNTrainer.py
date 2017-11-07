from AbstractTrainer import AbstractTrainer
from sklearn.neighbors import KNeighborsClassifier
class KNNTrainer(AbstractTrainer):

	def __init__(self, imageVectorList, imageLabelList, neighborCount = 5):
		"""

		:param neighborCount: No. of nearest neighbor to be considered
		"""
		if neighborCount > len(imageVectorList):
			raise ValueError("No. of nearest neighbors > no. of training records")
		self.neighborCount = neighborCount
		self.super(imageVectorList, imageLabelList)

	def buildModel(self, imageVectorList, imageLabelList):
		self.model = KNeighborsClassifier(self.neighborCount)
		self.model.fit(imageVectorList, imageLabelList)

