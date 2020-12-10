import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

class BagOfWords(object):
    """
    Class for implementing Bag of Words
     for Q1.1
    """
    def __init__(self, vocabulary_size):
        """
        Initialize the BagOfWords model
        """
        self.vocabulary_size = vocabulary_size

    def preprocess(self, text):
        """
        Preprocessing of one Review Text
            - convert to lowercase
            - remove punctuation
            - empty spaces
            - remove 1-letter words
            - split the sentence into words

        Return the split words
        """
        text = text.lower()
        bow = re.split(r'[.$\-\n\r& \",!/?_(#;\':)*]', text)
        bow = [word for word in bow if len(word) > 1] # remove empty strings and 1-letter words
        return bow

    def fit(self, X_train):
        X_train_text = X_train['Review Text']
        all_word_counts = Counter()
        for text in X_train_text:
            bow = self.preprocess(text)
            all_word_counts.update(bow)
        self.word_counts_ = all_word_counts.most_common(10)
        self.vocabulary_ = sorted(list(zip(*self.word_counts_))[0])
        self.word2idx_ = {word: idx for idx, word in enumerate(self.vocabulary_)}
        
    def transform(self, X):
        """
        Transform the texts into word count vectors (representation matrix)
            using the fitted vocabulary
        """
        output = []
        for text in X['Review Text']:
            bow = self.preprocess(text)
            bow = [word for word in bow if word in self.vocabulary_]
            word_counts = [0]*len(self.vocabulary_)
            for word in bow:
                word_counts[self.word2idx_[word]] += 1
            output.append(word_counts)
        return np.array(output)

class NaiveBayes(object):
    def __init__(self, beta=1, n_classes=2):
        """
        Initialize the Naive Bayes model
            w/ beta and n_classes
        """
        self.beta = beta
        self.n_classes = n_classes
    
    def preprocess(self, X):
        # Don't bother using any of the other features
        X = X.drop(['Title', 'Division Name', 'Department Name', 'Class Name', 'Age'], axis=1)

        # N x D matrix, N is the number of training instances, D is the vocab length
        N, D = X.shape[0], len(self.vocab_)
        doc_term_matrix = CountVectorizer(vocabulary=self.vocab_).fit_transform(X['Review Text']).toarray() # todo: dont convert to dense
        doc_term_matrix = pd.DataFrame(doc_term_matrix, columns=self.vocab_)

        X = X.drop('Review Text', axis=1)
        X = pd.concat([X, doc_term_matrix], axis=1)
        assert(X.shape == (N, D))
        return X

    def fit(self, X_train, y_train):
        """
        Fit the model to X_train, y_train
            - build the conditional probabilities
            - and the prior probabilities
        """
        self.vocab_ = sorted(CountVectorizer().fit(X_train['Review Text']).vocabulary_.keys())
        #self.word2idx_ = {word: idx for idx, word in enumerate(self.vocab_)}
        self.classes_ = sorted(set(y_train))
        assert(len(self.classes_) == self.n_classes)
        self.class_word_counts_ = {}
        self.class_total_word_counts_ = {}

        X_train = self.preprocess(X_train) # convert text to word counts + get rid of other features

        for kls in self.classes_:
            X_with_class = X_train[y_train == kls]
            word_counts_for_class = np.squeeze(X_with_class.sum(axis=0))
            assert(word_counts_for_class.shape == (len(self.vocab_),))

            self.class_word_counts_[kls] = word_counts_for_class
            self.class_total_word_counts_[kls] = word_counts_for_class.sum()
        
        self.total_word_count_ = sum([self.class_total_word_counts_[kls] for kls in self.classes_])
    
    def log_data_likelihood(self, X, kls):
        """
        log P(X|Y). (N,) where N is the number of instances.
        """
        map_counts = self.class_word_counts_[kls] + self.beta # (D,)
        map_total = self.class_total_word_counts_[kls] + self.beta*len(self.vocab_) # scalar
        log_word_likelihoods = np.log(map_counts / map_total) # (D,)
        return np.matmul(X, log_word_likelihoods) # (N, D) x (D,) = (N,)
    
    def log_prior(self, kls):
        """
        log P(Y). Scalar.
        """
        return np.log((self.class_total_word_counts_[kls] + self.beta*len(self.vocab_)) / (self.total_word_count_ + self.beta*len(self.vocab_)*len(self.classes_)))

    def predict(self, X_test):
        """
        Predict the X_test with the fitted model
        """
        X_test = self.preprocess(X_test)
        n_instances = X_test.shape[0]

        scores = pd.DataFrame(index=range(n_instances), columns=self.classes_)
        for kls in self.classes_:
            scores[kls] = self.log_prior(kls) + self.log_data_likelihood(X_test, kls)
        y_pred = scores.idxmax(axis=1)
        return y_pred

DATA_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'Data'))
FIGURES_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'Figures'))

def confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix of the
        predictions with true labels
    """
    classes = set(pd.concat([y_true, y_pred]))
    result = pd.DataFrame(index=classes, columns=classes)
    for class_true in classes:
        for class_pred in classes:
            result.loc[class_true, class_pred] = ((y_true == class_true) & (y_pred == class_pred)).sum() / len(y_true)
    return result

def load_data():
    """
    Load data

    Return
    ------
    X_train
    y_train
    X_valid
    y_valid
    X_test
    """
    fnames = ['X_train.csv', 'Y_train.csv', 'X_val.csv', 'Y_val.csv', 'X_test.csv']
    return [pd.read_csv(os.path.join(DATA_DIR, fname)) for fname in fnames]

def main():
    # Load in data
    X_train, y_train, X_valid, y_valid, X_test = load_data()
    test_ids = X_test['ID']
    # ids aren't relevant for learning, plus they're aligned between X/ys so we don't need to do any matchmaking
    for df in X_train, y_train, X_valid, y_valid, X_test:
        df.drop('ID', axis=1, inplace=True)
    y_train = y_train['Sentiment']
    y_valid = y_valid['Sentiment']

    # Fit the Bag of Words model for Q1.1
    print('Q1.1')
    bow = BagOfWords(vocabulary_size=10)
    print('Fitting BagOfWords model on first 100 training instances')
    bow.fit(X_train[:100])
    print('Transforming instances 101-200 with BagOfWords model')
    representation = bow.transform(X_train[100:200])
    print('Shape:', representation.shape)
    print(representation[:10])
    print('Word counts:')
    print(representation.sum(axis=0))
    print('Vocab:')
    print(bow.vocabulary_)
    print()

    # Fit the Naive Bayes model for Q1.3
    print('Q1.3')
    nb = NaiveBayes(beta=1)
    print('Fitting NaiveBayes model to training data')
    nb.fit(X_train, y_train)
    print('Making predictions with NaiveBayes model on validation data')
    y_pred = nb.predict(X_valid)

    y_pred = y_pred.replace('Positive', 1).replace('Negative', 0)
    y_valid = y_valid.replace('Positive', 1).replace('Negative', 0)

    print('Accuracy:', accuracy_score(y_valid, y_pred))
    print('Precision:', precision_score(y_valid, y_pred))
    print('Recall:', recall_score(y_valid, y_pred))
    print('ROC/AUC:', roc_auc_score(y_valid, y_pred))
    print('F1:', f1_score(y_valid, y_pred))

    print('Confusion matrix (axis=0 is true, axis=1 is pred):')
    print(confusion_matrix(y_valid, y_pred))
    print()

    print('Q1.4')
    beta_values = list(range(1, 11))
    scores = {}
    for beta in beta_values:
        print(f'NB: Testing beta={beta}')
        nb = NaiveBayes(beta=beta)
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_valid)
        
        y_pred = y_pred.replace('Positive', 1).replace('Negative', 0)
        y_valid = y_valid.replace('Positive', 1).replace('Negative', 0)

        scores[beta] = f1_score(y_valid, y_pred), roc_auc_score(y_valid, y_pred)
    best_beta = max(beta_values, key=lambda beta: scores[beta][0]) # optimize for best f1 on validation set
    print(f'Best value of beta is {best_beta} with validation scores f1: {scores[best_beta][0]} roc/auc: {scores[best_beta][1]}')
    print('Generating plot of roc/auc scores...')
    xs, ys = beta_values, [scores[beta][1] for beta in beta_values]
    plt.scatter(xs, ys)
    plt.xlabel('Beta')
    plt.ylabel('ROC/AUC score')
    plt.ylim(bottom=0., top=1.)
    plt.savefig(os.path.join(FIGURES_DIR, 'q_1_4_auc_vs_beta.png'))
    print('Done')
    print()

if __name__ == '__main__':
    main()
