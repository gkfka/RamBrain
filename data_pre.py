from pprint import pprint

from gensim.test.utils import common_texts
from gensim.models import Word2Vec

PADDING = '$'
def read_sent(file_name):
    tag_sent = []
    sent = []

    f = open(file_name, 'r', encoding='utf8').read()

    for i in f.split('\n'):
        ttm = [s[-1] for s in i.split()]
        stm = [s[0] for s in i.split()]

        tag_sent.append(ttm)
        sent.append(stm)

    return sent, tag_sent

def get_tirgram(sentence):
    sent = []
    for i in sentence:
        s = PADDING + ''.join(i) + PADDING
        tri = []
        # input(s)
        for j in range(len(s)-2):
            token = s[j]+s[j+1]+s[j+2]
            tri.append(token)
        sent.append(tri)
    return sent

from gensim.models import KeyedVectors

if __name__ == '__main__':
    file_name = "corpus/spacing_corpus.txt"

    sent, tag_sent = read_sent(file_name)
    # print((get_tirgram(sent)))

    print("word2ver...")
    model = Word2Vec(sentences=get_tirgram(sent), vector_size=128, window=5, min_count=0, workers=4)
    # model.save("word2vec.model")
    # model.wv.save_word2vec_format('model/embeding_model.bin')
    model.save('model/embeding_model.bin')
    print("done!")

