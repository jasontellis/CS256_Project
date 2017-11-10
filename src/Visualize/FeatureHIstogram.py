import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import pandas
from ImageFeatureExtractor import ImageFeatureExtractor, TrainingFileReader


class FeatureHistogram:


	def __init__(self, imageVectorList):
		colLabels = ImageFeatureExtractor.FEATURE_LABELS
		subPlotColumns = 3
		subPlotRows = len(colLabels) / subPlotColumns

		data = pandas.DataFrame.from_records(imageVectorList)

		data.plot(kind = 'box', subplots = True, layout = (subPlotRows, subPlotColumns), sharex = False, sharey = False)
		plt.show()


if __name__ == "__main__":
	TrainingFileReader
