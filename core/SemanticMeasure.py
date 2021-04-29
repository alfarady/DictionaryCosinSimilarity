from re import sub
import threading
import os
import json
from multiprocessing import cpu_count

import gensim.downloader as api
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity
from gensim.models.keyedvectors import Word2VecKeyedVectors
from gensim.models.fasttext import load_facebook_model
from core.IndoStopword import STOP_WORDS

nltk_stop_words = STOP_WORDS
DIR_DATASET = '/dataset'

class NotReadyError(Exception):
    pass

class SemanticMeasure:
    default_model = "../../cc.id.300.bin"
    model_ready = False
    corpus = []
    documents = list()
    content = list()
    dir_dataset = ''
    
    def __init__(self, stopwords=None, verbose=False):
        self.verbose = verbose
        self.dir_dataset = os.getcwd() + DIR_DATASET
        
        if self.verbose: 
            print('Init')

        # Load stopword
        if stopwords is None:
            self.stopwords = nltk_stop_words
        else:
            self.stopwords = stopwords

        # Load model
        self._load_model()
        self._prepare_data()

    def _load_model(self):
        # Pass through to _setup_model (overridden in threaded)
        self._setup_model()

    def _setup_model(self):
        
        if self.verbose: 
            print('Loading model')

        loaded_model = load_facebook_model(self.default_model)
        self.model = loaded_model.wv

        if self.verbose: 
            print('Model loaded')

        self.similarity_index = WordEmbeddingSimilarityIndex(self.model)
        self.model_ready = True

    def _prepare_data(self):
        if self.verbose: 
            print('Loading dataset')

        with open('dataset.json', encoding='utf-8') as in_file:
            dataset = json.load(in_file)

        self.documents = [item[0] for item in dataset['data']]
        self.content = [item[1] for item in dataset['data']]

        self.corpus = [self.preprocess(document) for document in self.documents]

        if self.verbose: 
            print('Dataset loaded')


    def preprocess(self, doc: str):
        # Clean up input document string, remove stopwords, and tokenize
        doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
        doc = sub(r'<[^<>]+(>|$)', " ", doc)
        doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
        doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
        
        return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in self.stopwords]

    def _softcossim(self, query: str, documents: list):
        # Compute Soft Cosine Measure between the query and each of the documents.
        query = self.tfidf[self.dictionary.doc2bow(query)]
        index = SoftCosineSimilarity(
            self.tfidf[[self.dictionary.doc2bow(document) for document in documents]],
            self.similarity_matrix)
        similarities = index[query]

        return similarities

    def similarity_query(self, query_string: str):
        if len(query_string) == 0:
            return False

        if self.model_ready:
            query = self.preprocess(query_string)

            if set(query) == set([word for document in self.corpus for word in document]):
                raise ValueError('query_string full overlaps content of document corpus')
            
            if self.verbose:
                print(f'{len(self.corpus)} documents loaded into corpus')
            
            self.dictionary = Dictionary(self.corpus+[query])
            self.tfidf = TfidfModel(dictionary=self.dictionary)
            self.similarity_matrix = SparseTermSimilarityMatrix(self.similarity_index, 
                                                self.dictionary, self.tfidf)
                        
            scores = self._softcossim(query, self.corpus)
            similarities = scores.tolist()

            res = list()

            for idx, score in (sorted(enumerate(similarities), reverse=True, key=lambda x: x[1])[:15]):
                if(score > 0):
                    temp = {}
                    temp['file'] = "file" + str(idx)
                    temp['keyword'] = self.documents[idx]
                    temp['content'] = self.content[idx]
                    temp['weight'] = score
                    res.append(temp)

            return res

        else:
            raise NotReadyError('Word embedding model is not ready.')

    def walid_similarity_query(self, answer: str, key: str):
        if len(answer) == 0 or len(key) == 0:
            return False

        if self.model_ready:
            documents = [answer, key]
            
            if self.verbose:
                print(f'{len(documents)} documents loaded and ready to preprocess')

            corpus = [self.preprocess(document) for document in documents]
            
            if self.verbose:
                print(f'{len(corpus)} documents loaded into corpus')
            
            dictionary = Dictionary(corpus)
            tfidf = TfidfModel(dictionary=dictionary)
            similarity_matrix = SparseTermSimilarityMatrix(self.similarity_index, dictionary, tfidf)

            answer_bow = dictionary.doc2bow(self.preprocess(answer))
            key_bow = dictionary.doc2bow(self.preprocess(key))
            
            # Measure soft cosine similarity
            scores = similarity_matrix.inner_product(answer_bow, key_bow, normalized=True)

            return scores

        else:
            raise NotReadyError('Word embedding model is not ready.')