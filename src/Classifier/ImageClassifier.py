
class ImageClassifier:
	'''
	Image classifier built using a model,
	classifies image passed to it
	'''
	def __init__(self, trainedModel):
		"""

		:param trainedModel: Model trained using a classifier
		"""
		self.trainedModel = trainedModel

	def classify(self, imageVector):
		'''

		:param imageVector: Feature Vector of Image
		:return: Predicted Label of Image
		'''
		classLabel = self.trainedModel.predict(imageVector)

		return classLabel
