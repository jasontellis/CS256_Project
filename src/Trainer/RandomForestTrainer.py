from sklearn.ensemble import RandomForestClassifier

from AbstractTrainer import AbstractTrainer


class RandomForestTrainer(AbstractTrainer):
	def __init__(self, imageVectorList, imageLabelList):
		super.__init__(imageVectorList, imageLabelList)
		self.model = RandomForestClassifier()
		self.model.fit(imageVectorList, imageLabelList)
