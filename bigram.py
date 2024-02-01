import sys
from pyspark import SparkContext, SparkConf
from operator import add

def preprocess_line(line):
    """
    Preprocess the line: lowercasing and removing non-alphabetic characters
    """
    import re
    # Convert to lower case and keep only alphabetic characters and spaces
    line = line.lower()
    line = re.sub('[^a-z\s]+', ' ', line)
    return line.split()  # Return a list of words instead of a string

def generate_bigrams(words_list):
    """
    Generate bigrams from a list of words
    """
    return [((words_list[i], words_list[i+1]), 1) for i in range(len(words_list)-1)]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: bigram_model <input_file> <output_dir>", file=sys.stderr)
        exit(-1)

    # Initialize Spark
    conf = SparkConf()
    sc = SparkContext(conf=conf)

    # Read data, preprocess each line, and split into words
    lines = sc.textFile(sys.argv[1])
    words_lists = lines.map(preprocess_line)

    # Generate bigrams and count occurrences of each bigram
    bigrams = words_lists.flatMap(generate_bigrams).reduceByKey(add)

    # Count occurrences of each word
    word_counts = words_lists.flatMap(lambda words_list: [(word, 1) for word in words_list]).reduceByKey(add)

    # Join the bigram counts with the first word counts to compute conditional frequencies
    bigram_with_first_word_count = bigrams.map(lambda x: (x[0][0], (x[0], x[1]))).join(word_counts)

    # Calculate the conditional probability for each bigram
    conditional_frequencies = bigram_with_first_word_count.map(lambda x: (x[1][0][0], x[1][0][1] / float(x[1][1])))

    # Save the output
    conditional_frequencies.coalesce(1, shuffle=True).saveAsTextFile(sys.argv[2])

    # Stop the Spark context
    sc.stop()
