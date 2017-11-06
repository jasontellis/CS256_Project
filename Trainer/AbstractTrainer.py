class AbstractTrainer:

	def __init__(self, imageDirectory):
		self.dataCount = 0
		self.imageDirectory = imageDirectory
		self.labels = []
		self.model = None

	def getLabels(self):
		"""
		Returns List of Binary Labels from Image Directory
		:return:
		"""

	def getModel(self):
		"""
		Returns trained model
		:return:
		"""
		return self.model