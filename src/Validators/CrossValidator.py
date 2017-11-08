from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


class CrossValidator:
	def __init__(self, classifier, imageVectorList, imageClassLabel, foldCount = 3):
		"""

		:param classifier:
		:param imageVectorList:
		:param imageClassLabel:
		:param foldCount:
		"""
		accuracy = cross_val_score(classifier.getModel(), imageVectorList, imageClassLabel, cv = foldCount,
		                           scoring = 'accuracy')
		self.predictions = cross_val_predict(classifier.getModel(), imageVectorList, imageClassLabel, cv = foldCount)
		self.classifierName = classifier.name
		self.accuracy = round(accuracy.mean(), 4)

	def getPredictions(self):
		return self.predictions

	def printAccuracy(self):
		print ("Accuracy of %s: %f" % (self.classifierName, self.accuracy))

	def getAccuracy(self):
		return self.accuracy
