from sklearn.ensemble import RandomForestClassifier

from AbstractTrainer import AbstractTrainer


class RandomForestTrainer(AbstractTrainer):
	def __init__(self, parameter = 10):
		AbstractTrainer.__init__(self, parameter)
		self.name = "Random Forest Classifier"
		self.model = RandomForestClassifier(n_estimators = parameter)
