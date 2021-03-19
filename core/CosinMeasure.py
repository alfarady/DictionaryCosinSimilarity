import nltk
import os
from math import log, sqrt
from collections import defaultdict

nltk.download('punkt')
DIR_DATASET = '/dataset'

class CosineMeasure():
    def __init__(self):
        self.nos_of_documents = 0
        self.vects_for_docs = []
        self.vect5_for_docs = {}
        self.document_freq_vect = {}
        self.dir_dataset = os.getcwd() + DIR_DATASET

        self.prepare_dataset()

    def make_prediction(self, query):
        if len(query) == 0:
            return False

        query_list = self.get_tokenized_and_normalized_list(query)
        query_vector = self.create_vector_from_query(query_list)
        self.get_tf_idf_from_query_vect(query_vector)
        result_set = self.get_result_from_query_vect(query_vector)

        parsed_data = []
        for item in result_set:
            if item[1] == 0:
                continue
            temp = {}
            temp["weight"] = item[1]
            temp["file"] = item[0]
            temp["content"] = open(self.dir_dataset + '/' + item[0], encoding="utf8").read()
            parsed_data.append(temp)

        return parsed_data
    
    def prepare_dataset(self):
        for file in os.listdir(self.dir_dataset):
            doc_text = self.get_document_text_from_doc_id(self.dir_dataset, file)
            token_list = self.get_tokenized_and_normalized_list(doc_text)
            vect = self.create_vector(token_list)
            self.vects_for_docs.append(vect)
            lineno = 1
            vect5 = defaultdict(list)
            lines = open(self.dir_dataset + '/' + file, encoding="utf8").read().splitlines()
            ps = nltk.stem.PorterStemmer()
            for line in lines:
                tokens = nltk.word_tokenize(line)
                for word in tokens:
                    vect5[ps.stem(word)].append(lineno)
                lineno = lineno+1;        
            self.vect5_for_docs[file] = vect5
            self.nos_of_documents = self.nos_of_documents + 1

    # Utils
    def create_vector_from_query(self, l1):
        vect = {}
        for token in l1:
            if token in vect:
                vect[token] += 1.0
            else:
                vect[token] = 1.0
        return vect

    def get_tf_idf_from_query_vect(self, query_vector1):
        vect_length = 0.0
        for word1 in query_vector1:
            word_freq = query_vector1[word1]
            if word1 in self.document_freq_vect:
                query_vector1[word1] = self.calc_tf_idf(word1, word_freq)
            else:
                query_vector1[word1] = log(1 + word_freq) * log(self.nos_of_documents)
            vect_length += query_vector1[word1] ** 2
        vect_length = sqrt(vect_length)
        if vect_length != 0:
            for word1 in query_vector1:
                query_vector1[word1] /= vect_length

    def calc_tf_idf(self, word1, word_freq):
        return log(1 + word_freq) * log(self.nos_of_documents / self.document_freq_vect[word1])

    def get_dot_product(self, vector1, vector2):
        if len(vector1) > len(vector2):
            temp = vector1
            vector1 = vector2
            vector2 = temp
        keys1 = vector1.keys()
        keys2 = vector2.keys()
        sum = 0
        for i in keys1:
            if i in keys2:
                sum += vector1[i] * vector2[i]
        return sum

    def get_tokenized_and_normalized_list(self, doc_text):
        tokens = nltk.word_tokenize(doc_text)
        ps = nltk.stem.PorterStemmer()
        stemmed = []
        for words in tokens:
            stemmed.append(ps.stem(words))
        return stemmed

    def create_vector(self, l1):
        vect = {}

        for token in l1:
            if token in vect:
                vect[token] += 1
            else:
                vect[token] = 1
                if token in self.document_freq_vect:
                    self.document_freq_vect[token] += 1
                else:
                    self.document_freq_vect[token] = 1
        return vect

    def get_document_text_from_doc_id(self, x,file):
        try:
            str1 = open(x+'/'+file, encoding="utf8").read()
        except:
            str1 = ""
        return str1

    def get_result_from_query_vect(self, query_vector1):
        parsed_list = []
        count=0;
        for file in os.listdir(self.dir_dataset):
            dot_prod = self.get_dot_product(query_vector1, self.vects_for_docs[count])
            parsed_list.append((file, dot_prod))
            parsed_list = sorted(parsed_list, key=lambda x: x[1])
            count=count+1
        return parsed_list

    