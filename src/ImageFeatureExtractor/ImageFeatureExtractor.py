import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class ImageFeatureExtractor:
	def __init__(self, imageFileName):
		self.feature = [1, 2, 3, 4]
		pass

	def getFeature(self):
		return self.feature


if __name__ == '__main__':
	pass
