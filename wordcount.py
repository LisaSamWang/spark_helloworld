import sys 
from pyspark import SparkContext, SparkConf

conf = SparkConf()
sc = SparkContext(conf=conf)
words = sc.textFile(sys.argv[1]).flatMap(lambda line: line.split(" ")) #read data from text file and split each line into words
wordCounts = words.map(lambda word: (word, 1)).reduceByKey(lambda a,b:a +b) #map each word to a key value pair and then reduce by key to get the count of each word
wordCounts.coalesce(1,shuffle=True).saveAsTextFile(sys.argv[2]) #save the output to another text file
sc.stop() #stop the spark context