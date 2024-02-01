import sys
from pyspark import SparkContext, SparkConf
from operator import add
import re

# Check if sufficient arguments are provided
if len(sys.argv) != 3:
    print("Usage: spark-submit script_name.py <input_file> <output_directory>")
    sys.exit(1)

input_file = sys.argv[1]
output_directory = sys.argv[2]

conf = SparkConf()
sc = SparkContext(conf=conf)

def preprocess_line(line):
    line = line.lower()
    line = re.sub(r'[^a-z\s]', '', line)
    return line

# Load and preprocess the text file
lines = sc.textFile(input_file).map(preprocess_line)

# Generate bigrams
bigrams = lines.flatMap(lambda line: [((line.split()[i], line.split()[i + 1]), 1) for i in range(len(line.split()) - 1)])

# Count the frequency of each bigram
bigram_counts = bigrams.reduceByKey(add)

# Count the frequency of each word
word_counts = lines.flatMap(lambda line: line.split()).map(lambda word: (word, 1)).reduceByKey(add)

# Calculate bigram probabilities
# Join the bigram_counts RDD with the word_counts RDD to get the probability of each bigram
# Each element in the bigram_counts RDD (Resilient Distributed Dataset) is a tuple of the form ((word1, word2), (bigram_count, word1_count))
# x[0] is the bigram (word1, word2), x[1] is (bigram_count, word1_count). The probability of the bigram is bigram_count/word1_count
bigram_probabilities = bigram_counts.join(word_counts).map(lambda x: (x[0], float(x[1][0])/x[1][1]))

# Save the results to a file
bigram_probabilities.coalesce(1, shuffle=True).saveAsTextFile(output_directory)

sc.stop()
