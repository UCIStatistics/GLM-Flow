from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext

# Based on
# https://github.com/apache/spark/blob/master/examples/src/main/python/mllib/logistic_regression_with_lbfgs_example.py

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

sc = SparkContext(appName="PythonLogisticRegressionWithLBFGSExample")
data = sc.textFile("iris.txt")
parsedData = data.map(parsePoint)

# Build the model
model = LogisticRegressionWithLBFGS.train(parsedData, regType=None)
print("Weights = " + str(model.weights))

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))

# Results:
# Weights = [0.749012177801,-1.89459008902,1.05730844301,-2.90734097089]
#Sepal.Length  Sepal.Width Petal.Length  Petal.Width 
#    0.749045    -1.894611     1.057191    -2.907072 
# success!
