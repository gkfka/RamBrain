from gensim.models import Word2Vec
from nltk.util import bigrams
from nltk.lm.preprocessing import pad_both_ends

def get_sents(fnm, size=-1):
    raw_sents = open(fnm, "r", encoding='utf8').read().split("\n")
    word_sents = []
    tag_sents = []

    if size>0 : raw_sents = raw_sents[:size]
    for s in raw_sents:
        s = s.split()
        words = [  _s[0] for _s in s]
        tags =  [ _s[-1] for _s in s]
        
        word_sents.append(words)
        tag_sents.append(tags)

    return word_sents, tag_sents

def get_bigram_sents(fnm, size=-1):
    raw_sents = open(fnm, "r", encoding='utf8').read().split("\n")
    word_sents = []
    tag_sents = []

    if size>0 : raw_sents = raw_sents[:size]
    for s in raw_sents:
        s = s.split()
        words = [  _s[0] for _s in s]
        tags =  [ _s[-1] for _s in s]
        
        data_list = list(pad_both_ends(words, n=2))
        bigram_sent = list(bigrams(data_list))
        bigram_sent = [ w[0]+w[1] for w in bigram_sent ]
        tags.append('I')

        word_sents.append(bigram_sent)
        tag_sents.append(tags)

    return word_sents, tag_sents

if __name__ == "__main__":
    w_sent, t_sent = get_bigram_sents("corpus/spacing_corpus.txt")
    print(w_sent)
    # print("Max sentence lenth : ", max(map(len, w_sent)))
    #
    # model = Word2Vec(w_sent, vector_size = 128, min_count=0) #128차원
    # print("Training End !")
    # #model.wv.save_word2vec_format('model/embeding_model.bin')
    # model.save('model/embeding_model.bin')