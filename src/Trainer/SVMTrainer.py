from sklearn.svm import SVC

from AbstractTrainer import AbstractTrainer


class SVMTrainer(AbstractTrainer):
	def __init__(self, imageVectorList, imageLabelList):
		"""

		:param imageVectorList:
		:param imageLabelList:
		"""
		super.__init__(imageVectorList, imageLabelList)
		self.model = SVC(kernel = 'linear')
		self.model.fit(imageVectorList, imageLabelList)
