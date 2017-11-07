class AbstractTrainer:
	"""
	Takes list of image vectors & associated labels as input & returns
	trained model as output
	"""

	def __init__(self, imageVectorList, imageLabelList):
		"""

		:param imageVectorList: List of Image Vectors
		:param imageLabelList: List of labels associated with list of image vectors
		"""
		self.model = None
		pass
		self.__validate__(imageVectorList, imageLabelList)
		self.buildModel()

	def __validate__(self, imageVectorList, imageLabelList):
		if len(imageLabelList) != len(imageVectorList):
			raise ValueError("Count of vectors and count of labels do not match")
		if not len(imageVectorList):
			raise ValueError("Data not provided for imageVectorList")
		if not len(imageLabelList):
			raise ValueError("Data not provided for imageLabelList")

	def getModel(self):
		"""
		Returns trained model
		:return:
		"""
		return self.model

	def buildModel(self):
		pass
