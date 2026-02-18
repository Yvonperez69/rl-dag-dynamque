import os
from collections import Counter

def count_word_frequency(file_path):
    try:
        with open(file_path, 'r') as file:
            text = file.read().lower()
            words = text.split()
            if not words or len(words) == 1: # Check for empty list of words and single word in it
                return {}  # Return an empty dictionary instead of None
            else:
                return Counter(words)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return {}

def main():
    file_path = os.path.join(os.getcwd(), 'text_file.txt')
    if not os.path.exists(file_path):  # Check if the file exists before opening it
        print(f"File {file_path} does not exist.")
        return

    word_freq = count_word_frequency(file_path)
    if word_freq:
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
            print(f"{word}: {freq}")

if __name__ == "__main__":
    main()