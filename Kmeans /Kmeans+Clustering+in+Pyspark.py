
# coding: utf-8

# # Introduction

# Identifying similar customers or users having similar patterns is one of the challenges faced in today's world. Segmenting or grouping such customers can lead to developing new strategies which are specically created to target these users. One such algorithm which can do clustering is called as K-means algorithm. This algorithm uses distance metrics to find distances between observations and group the similar observations. This is however just a short overview, but there is a lot of math involved in this algorithm.<br>
# <br>
# The objective of this project would be to cluster the household holds having similar power usage pattern so that the power companies can develop efficient strategies for them. Also any anamoly or users misusing the resources can also be detected. Such a segmenting or clustering can thus help the business in a variety of ways and hence improve the efficiency of the business model.<br>
# <br>
# We would be using the housing dataset obtained from UCI web repository. This dataset has more than 2 million records and represents the power consumpton patterns collected at a minute interval.

# # Motivation

# With this project we aim to demonstrate the power of machine learning on apache spark and how it can be used in developing a clustering algoritm which will cluster all the similar users. We aim to optimize the business model of the power companies by giving them the information about their users. We also aim to convert the numbers stored by the business into real insights and solid patterns about their users.

# # Design

# The design of the report can be split into following steps,<br>
# <br>
# 1. Importing libraries and creating spark session.<br>
# 2. Loading the data and pre-processing it.<br>
# 3. Explorartory analysis.<br>
# 4. Model building, optimizing and evaluating.<br>
# 5. Inferences.<br>

# ## Step 1: Importing libraries and creating spark session

# In the below steps we will load the required libararies and create a spark session

# In[1]:

#Loading libraries required in our code.
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.sql.types import DoubleType


# In[2]:

#Creating the spark session
if __name__ == "__main__":
    spark = SparkSession         .builder         .appName("Kmeans")         .getOrCreate()


# ## Step 2: Loading the data and pre-processing it.

# We will use spark sqlContext to load the data in pyspark. We have a colon delimited file and hence we will explicitly define the separator in the below code.<br>
# Spark sqlContext was not able to accurately infer the schema of the data, hence we manually defined the schema for each column in the dataset.

# In[3]:

#Loading the dataset
df = sqlContext.read.format('com.databricks.spark.csv').options(header='true',delimiter=';')    .load('/home/meaww/Downloads/household_power_consumption.txt')


# In[4]:

#Changing data types of the columns.
df=df.withColumn("Global_active_power", df.Global_active_power.cast(DoubleType()))
df=df.withColumn("Global_reactive_power", df.Global_reactive_power.cast(DoubleType()))
df=df.withColumn("Voltage", df.Voltage.cast(DoubleType()))
df=df.withColumn("Global_intensity", df.Global_intensity.cast(DoubleType()))
df=df.withColumn("Sub_metering_1", df.Sub_metering_1.cast(DoubleType()))
df=df.withColumn("Sub_metering_2", df.Sub_metering_2.cast(DoubleType()))
df=df.withColumn("Sub_metering_3", df.Sub_metering_3.cast(DoubleType()))


# In[5]:

#Removing NA records and unwanted columns 
df=df.na.drop()
df=df.drop('Date')
df=df.drop('Time')


# ## Step 3: Explorartory analysis

# We will do some descriptive analysis of the datain the below steps. These will include getting the number of records in the data, viewing first n records of the data, getting summary dtas for the data etc.

# In[6]:

#getting number of observations
df.count()


# In[7]:

#Viewing first 5 rows of the data
df.show(5)


# In[8]:

#Getting summary of the dataset/
df.describe().toPandas().transpose()


# In[9]:

#Viewing the schema of the dataset.
df.printSchema()


# ## Step 4: Model building, optimizing and evaluating.

# We would be building a kmeans model here, however there are some important points which should be considered before building th model. As mentioned in the introduction that kmeans uses distance meaures to find similar users. This assumes that all the columns have a same scale. If the scale is not same, it should be normalized so that they are on same scale and the distances measured can be compared across columns.<br>
# The model also takes input in dense vector format and hence proper conversions are also done. We will use a assempler to create the dense vector.<br>
# 

# In[10]:

#Assempling and creating a dense vector of inputs.
featuresUsed = df.columns
assembler = VectorAssembler(inputCols=featuresUsed, outputCol="features_unscaled")
assembled = assembler.transform(df)


# In[11]:

#Scaling and normalizing the data.
scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", withStd=True, withMean=True)
scalerModel = scaler.fit(assembled)
scaledData = scalerModel.transform(assembled)


# In[12]:

scaledData = scaledData.select("features")
scaledData.persist()


# In[13]:

#Viewing first 5 rows of scaled data
scaledData.head(5)


# Kmeans algorithm requires the number of clusters to preknown or to be assumed and finding can be done by some calculations. The algorithm calculates distance of the points from the initially randomly selected centroids. Then we will group and form a cluster of records which are closest to each other.<br>
# Then based on the new groups we get a new adjusted centroid and distances are calculated again. This process continues for multiple iterations and the end result are the clusters having minimum within sum of squared errors.<br>
# However finding optimum number of clusters is also a challenge. To solve this, we will build the kmeans model on multiple number of cluster values and find the one which has the optimum value of within sum of squared errors.

# In[14]:

#Building model for different cluster values
for i in xrange(2,18):
    kmeans = KMeans().setK(i).setSeed(1+i)
    model = kmeans.fit(scaledData)
    wssse = model.computeCost(scaledData)
    print("Within Set Sum of Squared Errors for " + str(i) + " clusters is: " + str(wssse))


# We can see from the above set of values that the wsse doesn't decrease much after 13 clusters. Hence we would choose our cluster count as 13. It can also be said that we are splitting our user base into 13 categories which can be inferred by looking at the records. It is also highly possible that these may not be the total number of categories and there might be more such categories. These can be identified by getting the understanding of the business domain and checking the wsse on more number of clusters. We will now build the final model on 13 clusters again and append the cluster value to each record.

# In[15]:

#Buildinfg final model and appending the predictions/categories
kmeans = KMeans().setK(13).setSeed(14)
model = kmeans.fit(scaledData)
transformed = model.transform(scaledData)


# In[16]:

#Viewing first 5 rows of the data
transformed.head(5)


# ## Step 5: Inferences

# We have now idnetified the optimum number of clusters and found out the categories of each user. In the below step we will look at the cluster centroids. Since we built 13 clusters we will have 13 cluster centers. Each cluster center will have dimensions equal to that of the input data. These values in a way represents the mean values of all features for each cluster. And any new observation having values close to these centers, will have the category assigned of the nearest cluster.

# In[17]:

centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)


# # Challenges faced

# Folowing were the challenges faced,<br>
# 1. Installing spark and integrating it with jupyter.<br>
# 2. Defining the problem statement and getting the data.<br>
# 3. Since the data was large, the compute time and compute was very intensive<br>
# 4. Getting the data in required format and model building<br>

# # Conclusion

# We were successfull in clustering the households based on the power consumption patterns. Also the business was made aware about insights which were not possible to get earlier. This unsupervised learning approach has thus helped in improving the efficiency of the business.
