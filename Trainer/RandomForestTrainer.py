from AbstractTrainer import AbstractTrainer
from sklearn.ensemble import RandomForestClassifier
class RandomForestTrainer(AbstractTrainer):

	def __init__(self, imageVectorList, imageLabelList):
		self.model = RandomForestClassifier()
		self.model.fit(imageVectorList, imageLabelList)
