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


	def __init__(self, imageVectorList, title ='Feature Distribution for Good Images'):
		colLabels = Constants.FEATURE_LABELS
		subPlotColumns = 1
		subPlotRows = len(colLabels) / subPlotColumns + len(colLabels) % subPlotColumns

		dataFrame = pandas.DataFrame.from_records(imageVectorList,columns = Constants.FEATURE_LABELS )#, index = Constants.FEATURE_LABELS)

		print 'Model: ',dataFrame.describe()
		rowCount, colCount = dataFrame.shape


		xlimits = []
		ylimits = []
		ymin = 0
		ymax = rowCount


		axes = dataFrame.plot(kind = 'hist',
		                      subplots = True,
		                      layout = (4, 2),
		                      sharex = False,
		                      sharey = False,
		                      title = title)

		for colIndex in range(colCount):
			col = dataFrame.iloc[:, colIndex]
			xmin = min(col)
			xmax = max(col)
			print colIndex/2, colIndex%2
			axis = axes[colIndex/2][colIndex%2]
			axis.set_xlim(xmin,xmax)
			axis.set_ylim(ymin,ymax)


if __name__ == "__main__":
	imageVectorList_good, imageLabelList_good = TrainingFileReader.extractTrainingData("../data/training/ling", False, 1)
	fh_good = FeatureHistogram(imageVectorList_good, title = 'Feature Distribution for Good Images')
	#fig.suptitle('Good image', fontsize = 12)
	# fig.savefig("../good_feature_histogram.png")
	imageVectorList_bad, imageLabelList_bad = TrainingFileReader.extractTrainingData("../data/training/ling", False, 0)
	fh_bad = FeatureHistogram(imageVectorList_bad, title = 'Feature Distribution for Bad Images')
	# #fig.suptitle('Bad image', fontsize = 12)
	# fig.savefig('../bad_feature_histogram.png')
