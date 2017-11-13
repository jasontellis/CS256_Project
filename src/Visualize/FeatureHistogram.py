import os
import sys


import matplotlib.pyplot as plt
import pandas

print sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
print sys.path
from Constants import Constants
from TrainingFileReader  import TrainingFileReader
print Constants.CLASS_LABELS


class FeatureHistogram:


	def __init__(self, imageVectorList):
		colLabels = Constants.FEATURE_LABELS
		subPlotColumns = 3
		subPlotRows = len(colLabels) / subPlotColumns + len(colLabels) % subPlotColumns

		data = pandas.DataFrame.from_records(imageVectorList,columns = Constants.FEATURE_LABELS )#, index = Constants.FEATURE_LABELS)
		print data.describe()
		data.plot(kind = 'hist', subplots = True, layout = (4, 2), sharex = False, sharey = False)
		plt.show()


if __name__ == "__main__":
	imageVectorList, imageLabelList = TrainingFileReader.extractTrainingData("../data/training/ling/")
	fh = FeatureHistogram(imageVectorList)
