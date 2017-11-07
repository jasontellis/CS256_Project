from sklearn.model_selection import KFold

class CrossValidator:

	def __init__(self, foldCount = 3):
		kf = KFold(n_splits = foldCount)

