import sys
import math
from collections import Counter

def calculate_entropy(labels):
    total = len(labels)
    if total == 0.0:
        return 0.0
    counts = Counter(labels)
    entropy = 0.0
    for label in counts:
        prob = counts[label] / total
        entropy -= prob * math.log2(prob)

    return entropy

def calculate_error(labels):
    total = len(labels)
    if total == 0.0:
        return 0.0
    counts = Counter(labels)
    most_common_count = counts.most_common(1)[0][1]
    error = (total - most_common_count) / total
    
    return error

def main(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]

    header = lines[1]
    labels = [line[-1] for line in lines[1:]]

    entropy = calculate_entropy(labels)
    error = calculate_error(labels)

    with open(output_file, 'w') as f:
        f.write("Entropy: {}\n".format(entropy))
        f.write("Error: {}\n".format(error))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 inspection.py <input_file> <output_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)