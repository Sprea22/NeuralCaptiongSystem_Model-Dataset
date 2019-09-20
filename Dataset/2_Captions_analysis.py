import pandas as pd
import numpy as np

v0_captions_collection = pd.read_excel("Captions collection/v0_captions_collection.xlsx")

captions_numb_chars = []
captions_numb_words = []

for orig_cap in v0_captions_collection["caption_content"].values:
    captions_numb_chars.append(len(orig_cap))
    captions_numb_words.append(len(orig_cap.split(" ")))

avg_numb_chars = np.mean(captions_numb_chars)
avg_numb_words = np.mean(captions_numb_words)

std_numb_chars = np.std(captions_numb_chars)
std_numb_words = np.std(captions_numb_words)

print("AVG Number of chars per caption: ", round(avg_numb_chars, 2), " - STD: ", round(std_numb_chars, 2))
print("AVG Number of words per caption: ", round(avg_numb_words, 2), " - STD: ", round(std_numb_words, 2))
print("\n")
####

v1_captions_collection = pd.read_excel("Captions collection/v1_captions_collection.xlsx")
subsequence_numb_chars = []
subsequence_numb_words = []

for orig_cap in v1_captions_collection["caption"].values:
    subsequence_numb_chars.append(len(orig_cap))
    subsequence_numb_words.append(len(orig_cap.split(" ")))

avg_numb_chars = np.mean(subsequence_numb_chars)
avg_numb_words = np.mean(subsequence_numb_words)

std_numb_chars = np.std(subsequence_numb_chars)
std_numb_words = np.std(subsequence_numb_words)

print("AVG Number of chars per subsequence: ", round(avg_numb_chars, 2), " - STD: ", round(std_numb_chars, 2))
print("AVG Number of words per subsequence: ", round(avg_numb_words, 2), " - STD: ", round(std_numb_words, 2))
print("\n")