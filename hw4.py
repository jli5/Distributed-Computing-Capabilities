from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.feature import StandardScaler
import pyspark.mllib
import pyspark.mllib.regression
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import *
from numpy import array
from pyspark.sql import Row
import os
import shutil

# Codes reference: http://www.techpoweredmath.com/spark-dataframes-mllib-tutorial/

#Checking if results file exists,deleting file if it does.
if os.path.exists('/Users/jingli/Documents/Distributed-Computing-Capabilities/predicted_results'):
	shutil.rmtree('/Users/jingli/Documents/Distributed-Computing-Capabilities/predicted_results')
	


sc = SparkContext("local", "Simple App")



#Building the model using training data from Boston house csv
houses = sc.textFile('/Users/jingli/Documents/Distributed-Computing-Capabilities/boston_house.csv')
houses = houses.map(lambda line: line.split(","))
header = houses.first()
headerless_houses = houses.filter(lambda line: line != header)  



newRDD = headerless_houses.map(lambda x: Row(CRIM=x[0], ZN=x[1], INDUS=x[2], CHAS=x[3], NOX=x[4], RM=x[5], AGE=x[6], DIS=x[7], RAD=x[8], TAX=x[9], PTRATIO=x[10], B=x[11], LSTAT=x[12], AA_MEDV=x[13]))
features = newRDD.map(lambda row: row[1:])
features.take(5)

scaler = StandardScaler(withMean=True, withStd=True).fit(features)
features_transform = scaler.transform(features)
features_transform.take(5)

lab = newRDD.map(lambda row: row[0])
transformedData = lab.zip(features_transform)
transformedData = transformedData.map(lambda row: LabeledPoint(row[0],[row[1]]))

model = LinearRegressionWithSGD.train(transformedData, intercept=True)


########Using test data to predict values by applying model

verify = sc.textFile('/Users/jingli/Documents/Distributed-Computing-Capabilities/verification.csv')


verify = verify.map(lambda line: line.split(","))
header = verify.first()
headerless_verify = verify.filter(lambda line: line != header)
testRDD = headerless_verify.map(lambda x: Row(CRIM=x[0], ZN=x[1], INDUS=x[2], CHAS=x[3], NOX=x[4], RM=x[5], AGE=x[6], DIS=x[7], RAD=x[8], TAX=x[9], PTRATIO=x[10], B=x[11], LSTAT=x[12]))
features = testRDD.map(lambda row: row[0:])
features_transform = scaler.transform(features)
x= model.predict(features_transform)
print x.take(5)
x.saveAsTextFile('/Users/jingli/Documents/Distributed-Computing-Capabilities/predicted_results');
