import os

with open('data/en-fr/preprocessed/train.en' ,'r') as f:
    a = f.readline()
    print(a)


a = os.path.join('data/en-fr/preprocessed/BPE', 'bpe.codes.' + 'en')
print(a)

