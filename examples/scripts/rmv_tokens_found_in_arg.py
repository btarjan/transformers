
import sys

vocab_set = set()
vocab = open(sys.argv[1], "r")

for line in vocab:
    line = line.rstrip()    
    vocab_set.add(line)

for line in sys.stdin:
    line = line.rstrip()    
    word = line.split()[1]
    if word not in vocab_set:
        print(line)
