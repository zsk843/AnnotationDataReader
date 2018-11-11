import pickle
import os
import numpy as np
import string
import re
import shutil

from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet as wn


class TextData(object):
    PHENOTYPE = {"homeless": 0, "retired": 1, "unemployed": 2, "bowels": 3, "bladder": 4, "grooming": 5, "toilet": 6,
                 "feeding": 7, "transfer": 8, "mobility": 9, "dressing": 10, "stairs": 11, "bathing": 12,
                 "education": 13, "insurance": 14}
    STOP_WORDS = {"the", "that", "this", "a", "an", "will", "would", "should", "could", "might"}

    def __init__(self):
        self.text_dic = {}
        self.text_num = 0
        self.text_lst = []
        self.item_len = 0
        self.path = ""
        self.label_lst = None

    def load_from_raw(self, path):
        self.text_dic = {}
        self.text_num = 0
        self.text_lst = []
        self.item_len = len(TextData.PHENOTYPE)
        self.path = path
        for root, dirs, files in os.walk(path):
            for f in files:
                if str(f).endswith(".txt"):
                    self.text_dic[f.replace(" ", "")] = self.text_num
                    self.text_num += 1
                    with open(os.path.join(root, f), encoding="UTF-8") as fptr:
                        self.text_lst.append(fptr.read().lower())

        self.label_lst = np.zeros(len(self.text_dic) * self.item_len, dtype=np.int8).reshape((len(self.text_dic), -1))
        for root, dirs, files in os.walk(path):
            for f in files:
                if not (str(f).endswith(".txt") or str(f).endswith(".text") or str(f).endswith(".npy")):
                    with open(os.path.join(root, f), "rb") as fptr:
                        tmp_dic = pickle.load(fptr)
                        for file in tmp_dic:
                            name = os.path.basename(file)
                            label_index = int(self.text_dic[name.replace(" ", "")])
                            label_dic = eval(tmp_dic[file])
                            for label in label_dic:
                                if label in TextData.PHENOTYPE:
                                    self.label_lst[label_index][TextData.PHENOTYPE[label]] = int(label_dic[label])

    def load_from_exists(self, path):
        self.label_lst = None
        text_path = os.path.join(path, "text")
        files = os.listdir(text_path)
        self.text_lst = ["" for i in range(len(files))]
        for f in files:
            with open(os.path.join(text_path,f), 'r',encoding="UTF-8") as fptr:
                self.text_lst[int(f)] = fptr.read()
        self.label_lst = np.load(os.path.join(path, "label.npy"))

    @staticmethod
    def del_file(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            if os.path.isfile(path_file):
                os.remove(path_file)
            else:
                TextData.del_file(path_file)
        shutil.rmtree(path)

    def text_processing(self, fun_lst):
        for i in tqdm(range(len(self.text_lst))):
            tmp_txt = self.text_lst[i]
            for fun in fun_lst:
                tmp_txt = fun(tmp_txt)
            self.text_lst[i] = tmp_txt

    def label_processing(self, fun_lst):
        for i in range(int(self.label_lst.shape[0])):
            for fun in fun_lst:
                self.label_lst[i] = fun(self.label_lst[i])

    @staticmethod
    def rule_employment(vec):
        if vec[TextData.PHENOTYPE["retired"]] != 1 and vec[TextData.PHENOTYPE["unemployed"]] == 1:
            vec[TextData.PHENOTYPE["retired"]] = 0
        return vec

    @staticmethod
    def rule_mobility(vec):
        if vec[TextData.PHENOTYPE["stairs"]] <= 0 and vec[TextData.PHENOTYPE["mobility"]] == 0:
            vec[TextData.PHENOTYPE["stairs"]] = 0
        return vec

    @staticmethod
    def remove_punctuation(text):
        punctuation = string.punctuation.replace("[", "").replace("]", "") + string.digits
        table = str.maketrans('', '', punctuation)
        res = str(text).translate(table)
        return res

    @staticmethod
    def replace_hidden_words(text):
        out = re.sub("(\[.*?\])", "[hidden]", text)
        return out

    @staticmethod
    def remove_stop_words(text):
        word_tokens = word_tokenize(text)
        out = ""
        for w in word_tokens:
            if w not in TextData.STOP_WORDS:
                out = out + w + " "
        return out

    @staticmethod
    def stem_words(text):
        tokens = word_tokenize(text)
        # stemming of words
        from nltk.stem.porter import PorterStemmer
        porter = PorterStemmer()
        out = ""
        for w in tokens:
            out = out + porter.stem(w) + " "
        return out

    @staticmethod
    def original_form(text):
        tags = nltk.pos_tag(word_tokenize(text))
        out = ""
        for tag in tags:
            wn_tag = penn_to_wn(tag[1])
            out = out + WordNetLemmatizer().lemmatize(tag[0], wn_tag) + " "
        return out

    def save(self):
        save_path = os.path.join(self.path, "output")
        label_path = os.path.join(save_path, "label.npy")
        text_path = os.path.join(save_path, "text")
        if os.path.exists(save_path):
            TextData.del_file(save_path)

        os.makedirs(text_path)
        index = 0
        for text in self.text_lst:
            with open(os.path.join(text_path, str(index) + ".text"), "w", encoding="UTF-8") as f:
                f.write(text)
            index += 1

        np.save(label_path, self.label_lst)


def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return wn.NOUN
