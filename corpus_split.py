import random
import re
f = open("corpus/spacing_corpus.txt", "r", encoding="UTF-8")
sents = f.read().split("\n")
f.close()

f_train = open("corpus/corpus_train.txt", "w+", encoding="UTF-8")
f_test = open("corpus/corpus_test.txt", "w+", encoding="UTF-8")

# train : 50000, test : 10000
train_cnt, test_cnt = 50000, 10000
random.shuffle(sents)
print("\n".join(sents[:train_cnt]), file=f_train)
print("\n".join(sents[train_cnt:train_cnt+test_cnt]), file=f_test)

f_train.close()
f_test.close()
