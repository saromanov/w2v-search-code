import gensim
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import nltk
from nltk.corpus import stopwords


class W2vSearch:
    def __init__(self, data):
        ''' data provides training data as string or as list of strings
        '''
        links = data
        if type(data) is str:
            links = [data]
        self.model = gensim.models.Word2Vec(self._load(links), min_count=10)

    def _load(self, links):
        result = []
        for link in links:
            data = open(link, 'r').read()
            result.extend(self._preprocess(data.decode('utf-8')))
        return result

    def _preprocess(self, str):
        sents = [word_tokenize(" ".join(re.findall(r'\w+',t, flags=re.UNICODE | re.LOCALE)).lower())
            for t in sent_tokenize(str.replace("'",""))]
        stop = stopwords.words('english')
        for s in range(len(sents)):
            sents[s] = [w for w in sents[s] if w.lower() not in stop and len(w.lower()) > 2]
        return sents

    def similarity(self, word1, word2):
        ''' similarity returns similarity between two words '''
        return self.model.similarity(word1, word2)

    def search(self, word, negative=[]):
        return self.model.most_similar(positive=[word], negative=negative)

    def save(self, outfile):
        ''' This method provides saving of this model
        '''
        self.model.save(outfile)


def preprocessing(str):
    sents = [word_tokenize(" ".join(re.findall(r'\w+',t, flags=re.UNICODE | re.LOCALE)).lower())
        for t in sent_tokenize(str.replace("'",""))]
    stop = stopwords.words('english')
    for s in range(len(sents)):
        sents[s] = [w for w in sents[s] if w.lower() not in stop and len(w.lower()) > 2]
    return sents