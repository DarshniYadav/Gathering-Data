# Gathering-Data
# Creating Small scale chatGPT model for language translation: gathering data

In the realm of natural language processing (NLP), language translation remains one of the fundamental challenges. With advancements in deep learning, particularly with models like GPT (Generative Pre-trained Transformer), we can develop sophisticated language translation systems even on a small scale. In this article, we'll embark on a journey to create a small-scale chatGPT model for language translation using TensorFlow.
# Gathering Data
Data is the cornerstone of any machine learning project. For our translation model, we'll use the French-English dataset from TensorFlow's official repository. We'll download and extract the data using TensorFlow's utility functions.
import tensorflow as tf
# Gathering data using TensorFlow's utility function
text_file = tf.keras.utils.get_file(
    fname='fra-eng.zip',
    origin="http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip",
    extract=True
)

# Data Preprocessing
Once we have the data, the next step is preprocessing. We'll normalize the text, handle special characters, and format it appropriately for training.
# Importing necessary libraries
import pathlib
import unicodedata
import re

# Defining the path to the text file
text_file = pathlib.Path(text_file).parent / 'fra.txt'

def normalize(line):
    # Normalize unicode characters, strip leading/trailing whitespace, convert to lowercase
    line = unicodedata.normalize("NFKC", line.strip().lower())
    # Handle special characters and add start and end tokens for the target language (French)
    line = re.sub(r"^([^ \w])(?!\s)", r"\1", line)
    line = re.sub(r"(\s[^ \w])(?!\s)", r"\1", line)
    line = re.sub(r"(?!\s)([^ \w])$", r"\1", line)
    line = re.sub(r"(?!\s)([^ \w]\s)", r"\1", line)
    eng, fre = line.split("\t")
    fre = '[start] ' + fre + ' [end]'
    return eng, fre

# Read and normalize the text pairs
with open(text_file) as fp:
    text_pairs = [normalize(line) for line in fp]

# Tokenization and Statistics
Before training, it's crucial to tokenize our text data and understand its characteristics, such as the vocabulary size and maximum sequence lengths
# Initialize sets to store unique tokens for English and French
eng_tokens, fre_tokens = set(), set()
# Initialize variables to store maximum sequence lengths
eng_maxlen, fre_maxlen = 0, 0

# Iterate through text pairs to tokenize and compute statistics
for eng, fre in text_pairs:
    eng_token, fre_token = eng.split(), fre.split()
    eng_maxlen = max(eng_maxlen, len(eng_token))
    fre_maxlen = max(fre_maxlen, len(fre_token))
    eng_tokens.update(eng_token)
    fre_tokens.update(fre_token)

# Print statistics
print(f"Total tokens in English: {len(eng_tokens)}")
print(f"Total tokens in French: {len(fre_tokens)}")
print(f"Maximum length of English sequence: {eng_maxlen}")
print(f"Maximum length of French sequence: {fre_maxlen}")

# Data Serialization
Lastly, we'll serialize our preprocessed data for future use.
import pickle

# Serialize preprocessed data for future use
with open("text_pairs.pickle", 'wb') as fp:
    pickle.dump(text_pairs, fp)

# Conclusion
In this article, we've laid the groundwork for building a small-scale chatGPT model for language translation. We've gathered and preprocessed the data, tokenized it, and serialized i
