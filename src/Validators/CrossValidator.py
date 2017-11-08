from sklearn.model_selection import cross_val_score


class CrossValidator:
	def __init__(self, classifier, imageVectorList, imageClassLabel, foldCount = 5):
		"""

		:param classifier:
		:param imageVectorList:
		:param imageClassLabel:
		:param foldCount:
		"""
		accuracy = cross_val_score(classifier.getModel(), imageVectorList, imageClassLabel, cv = foldCount,
		                           scoring = 'accuracy')
		self.classifierName = classifier.name
		self.accuracy = round(accuracy.mean(), 4)

	def printAccuracy(self):
		print ("Accuracy of %s: %f" % (self.classifierName, self.accuracy))

	def getAccuracy(self):
		return self.accuracy
