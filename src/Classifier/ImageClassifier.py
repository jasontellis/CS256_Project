class ImageClassifier:
	def __init__(self, trainedModel):
		"""

		:param trainedModel: Model trained using a classifier
		"""
		self.trainedModel = trainedModel

	def classify(self, imageVector):
		classLabel = self.trainedModel.predict(imageVector)
		return classLabel
