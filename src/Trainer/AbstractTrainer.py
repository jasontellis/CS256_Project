import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class AbstractTrainer:
	"""
	Takes list of imageFile vectors & associated labels as input & returns
	trained model as output
	"""

	def __init__(self, parameter = None):
		"""

		:param parameter:
		"""
		self.name = ""
		self.parameter = parameter
		self.model = None

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

	def buildModel(self, imageVectorList, imageLabelList):
		"""
		Build the model using classifier

		:param imageVectorList:
		:param imageLabelList:
		:return:
		"""
		self.__validate__(imageVectorList, imageLabelList)
		self.model.fit(imageVectorList, imageLabelList)
		return self.model
