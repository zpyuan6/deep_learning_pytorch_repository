import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from collections import Counter
import json
from gensim.models import Word2Vec
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch

all_csv_path = "F:\cyberbullying\cyberbullying_tweets_clean_all.csv"
train_csv_path = "F:\cyberbullying\cyberbullying_tweets_clean_training_data.csv"
val_csv_path = "F:\cyberbullying\cyberbullying_tweets_clean_val_data.csv"
test_csv_path = "F:\cyberbullying\cyberbullying_tweets_clean_test_data.csv"
twee_type = ["not_cyberbullying","religion","age","ethnicity","gender","other_cyberbullying"]

EMBEDDING_DIM = 200

def Tokenize(column, vocab_to_int, seq_len):
    ##Tokenize the columns text using the vocabulary
    text_int = []
    for text in column:
        r = [vocab_to_int[word] for word in text.split()]
        text_int.append(r)

    ##Add padding to tokens
    features = np.zeros((len(text_int), seq_len), dtype = int)
    for i, review in enumerate(text_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[: seq_len]
        features[i, :] = np.array(new)

    return features


def create_dir(column):
    ##Create vocabulary of words from column
    corpus = [word for text in column for word in text.split()]
    count_words = Counter(corpus)
    # '''List the n most common elements and their counts from the most common to the least.  If n is None, then list all element counts.
    sorted_words = count_words.most_common()
    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
    
    vocab_json = json.dumps(vocab_to_int)
    f = open("vocabulary_dir.json", 'w')
    f.write(vocab_json)
    f.close()

    return vocab_to_int


def load_data():
    data_all = pd.read_csv(all_csv_path, encoding="utf-8")
    train_data = pd.read_csv(train_csv_path, encoding="utf-8")
    val_data = pd.read_csv(val_csv_path, encoding="utf-8")
    test_data = pd.read_csv(test_csv_path, encoding="utf-8")

    max_len = np.max(data_all['text_len'])

    # print(max_len)

    # train_data.info()
    # print(train_data.type.value_counts())
    # print("=================================================")
    
    # val_data.info()
    # print(val_data.type.value_counts())
    # print("=================================================")
    
    # test_data.info()
    # print(test_data.type.value_counts())
    # print("=================================================")
    # data.drop(data.columns[[0]])

    x_train = train_data['text_clean']
    y_train = train_data['type']

    x_val= val_data['text_clean']
    y_val = val_data['type']

    x_test = test_data['text_clean']
    y_test = test_data['type']

    

    dir = create_dir(data_all["text_clean"])

    x_train_token = Tokenize(x_train,dir,max_len)
    x_val_token = Tokenize(x_val,dir,max_len)
    x_test_token = Tokenize(x_test,dir,max_len)

    Word2vec_train_data = list(map(lambda x: x.split(), x_train))

    word2vec_model = Word2Vec(Word2vec_train_data, vector_size=EMBEDDING_DIM)
    print(f"Vocabulary size: {len(dir) + 1}")
    VOCAB_SIZE = len(dir) + 1 

    #define empty embedding matrix
    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    #fill the embedding matrix with the pre trained values from word2vec
    #corresponding to word (string), token (number associated to the word)
    for word in dir:
        if word2vec_model.wv.__contains__(word):
            embedding_matrix[dir[word]] = word2vec_model.wv.__getitem__(word)

    np.save("embedding_word", embedding_matrix)

    ros = RandomOverSampler() # 重采样除多数类以外的所有类；
    x_train_token_os, y_train_os = ros.fit_resample(np.array(x_train_token), np.array(y_train))
    # train_os = pd.DataFrame(list(zip([x[0] for x in x_train], y_train)), columns = ['text_clean', 'type'])
    (unique, counts) = np.unique(y_train_os, return_counts=True)
    print(np.asarray((unique, counts)).T)

    train_data = TensorDataset(torch.from_numpy(np.array(x_train_token_os)), torch.from_numpy(np.array(y_train_os)))
    val_data = TensorDataset(torch.from_numpy(np.array(x_val_token)), torch.from_numpy(np.array(y_val)))
    test_data = TensorDataset(torch.from_numpy(np.array(x_test_token)), torch.from_numpy(np.array(y_test)))

    print(f"train: {train_data.__len__()},val: {val_data.__len__()},test: {test_data.__len__()}")

    return train_data, val_data, test_data, VOCAB_SIZE

def main():
    train_data, val_data, test_data, a = load_data()



if __name__ == "__main__":
    main()

