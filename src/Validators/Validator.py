from sklearn.model_selection import cross_val_score


class CrossValidator:
	def __init__(self, trainedModel, imageVectorList, imageClassLabels, foldCount = 5):
		accuracy = cross_val_score(trainedModel, imageVectorList, imageClassLabels, cv = foldCount,
		                           scoring = 'accuracy_score')
		self.accuracy = round(accuracy.mean(), 4)

	def getAccuracy(self):
		return self.accuracy
