from AbstractTrainer import AbstractTrainer


class KNNTrainer(AbstractTrainer):
	def __init__(self, imageVectorList, imageLabelList, neighborCount = 5):
		"""

		:param neighborCount: No. of nearest neighbor to be considered
		"""

		super.__init__(imageVectorList, imageLabelList)
		if neighborCount > len(imageVectorList):
			raise ValueError("No. of nearest neighbors > no. of training records")
		self.neighborCount = neighborCount
		super.__init__(imageVectorList, imageLabelList)

	def buildModel(self):
		pass

	# self.model = KNeighborsClassifier(self.neighborCount)
	# self.model.fit(imageVectorList, imageLabelList)
