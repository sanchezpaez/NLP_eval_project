# Sandra Sanchez
# NLP Eval project
import random
from collections import Counter

from timeit import default_timer as timer
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from sklearn_crfsuite import CRF
import sklearn_crfsuite
from pathlib import Path
from conllu import parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.tag import hmm, crf, perceptron
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn_crfsuite import metrics as metrics_crf
from yellowbrick.model_selection import CVScores
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def read_datasets(corpus_dirname, train_filename, dev_filename, test_filename):
    CORPUS_PATH = Path(corpus_dirname)

    TRAIN_PATH = CORPUS_PATH / train_filename
    VAL_PATH = CORPUS_PATH / dev_filename
    TEST_PATH = CORPUS_PATH / test_filename

    with open(TRAIN_PATH, 'r') as file:
        train_data = parse(file.read())

    with open(VAL_PATH, 'r') as file:
        val_data = parse(file.read())

    with open(TEST_PATH, 'r') as file:
        test_data = parse(file.read())

    return train_data, val_data, test_data


def tag_datasets(train_set, val_set, test_set):
    tagged_sents_train = [[(word['form'], word['upos']) for word in sent] for sent in train_set]
    tagged_sents_val = [[(word['form'], word['upos']) for word in sent] for sent in val_set]
    tagged_sents_test = [[(word['form'], word['upos']) for word in sent] for sent in test_set]

    return tagged_sents_train, tagged_sents_val, tagged_sents_test


def get_sentences_from_datasets(train_set, val_set, test_set):
    sents_train = [[word['form'] for word in sent] for sent in train_set]
    sents_val = [[word['form'] for word in sent] for sent in val_set]
    sents_test = [[word['form'] for word in sent] for sent in test_set]

    return sents_train, sents_val, sents_test


def get_sentence_gold_labels_from_datasets(tagged_train, tagged_val, tagged_test):
    gold_sent_labels_train = [[(word[1]) for word in sent] for sent in tagged_train]
    gold_sent_labels_val = [[(word[1]) for word in sent] for sent in tagged_val]
    gold_sent_labels_test = [[(word[1]) for word in sent] for sent in tagged_test]

    return gold_sent_labels_train, gold_sent_labels_val, gold_sent_labels_test


def get_tokens_from_sentences(sent_train, sent_val, sent_test):
    tokens_train = [token for sentence in sent_train for token in sentence]
    tokens_val = [token for sentence in sent_val for token in sentence]
    tokens_test = [token for sentence in sent_test for token in sentence]
    return tokens_train, tokens_val, tokens_test


def get_token_gold_labels(gold_sent_labels_train, gold_sent_labels_val, gold_sent_labels_test):
    gold_tokens_train = [label for sentence in gold_sent_labels_train for label in sentence]
    gold_tokens_val = [label for sentence in gold_sent_labels_val for label in sentence]
    gold_tokens_test = [label for sentence in gold_sent_labels_test for label in sentence]
    return gold_tokens_train, gold_tokens_val, gold_tokens_test


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label in sent]


def sent2tokens(sent):
    return [token for token, label in sent]


def train_baseline(tokens_train, gold_train, tokens_val, gold_val, strategy):
    if strategy == 'most_frequent':
        classifier = DummyClassifier(strategy='most_frequent')
        classifier.fit(tokens_train, gold_train)
        multiclass_predictions = classifier.predict(tokens_val)
        multiclass_imbalanc_probs = classifier.predict_proba(tokens_val)
        accuracy = classifier.score(tokens_val, gold_val)
        print(f'Accuracy on the val set with strategy {strategy}: {accuracy}')
    elif strategy == 'mlp':
        classifier = MLPClassifier(verbose=True)
        X_train, y_train, X_val = transform_raw_data_into_matrix(tokens_train, gold_train, tokens_val)
        classifier.fit(X_train, y_train)
        multiclass_predictions = classifier.predict(X_val)
        multiclass_imbalanc_probs = classifier.predict_proba(X_val)
        accuracy = classifier.score(X_val, gold_val)
        print(f'Accuracy on the val set with strategy {strategy}: {accuracy}')

    return accuracy


def transform_raw_data_into_matrix(train_tokens, train_labels, dev_tokens):
    assert len(train_tokens) == len(train_labels)
    # print(sentences)
    y = np.array(train_labels)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_tokens)
    X_dev = vectorizer.transform(dev_tokens)
    return X_train, y, X_dev


def train_model(tagged_train_sents, sentences_val, tagged_val_sents):
    tagger = crf.CRFTagger()
    tagger.train(tagged_train_sents, 'model.crf.tagger')
    model_outputs = tagger.tag_sents(sentences_val)
    print(f"Predicted tagged val sentence {model_outputs[0]}")

    accuracy = tagger.evaluate(tagged_val_sents)
    print(f"Accuracy on the validation set with {str(tagger)}:", accuracy)  # 0.9793798916315473
    predicted_labels = [tag for sentence in model_outputs for token, tag in sentence]
    # predicted_labels = [[tag for token, tag in sent] for sent in model_outputs] # for sentence accuracy, but it does not work

    return model_outputs, tagger, accuracy, predicted_labels


def tune_hyperparameters(tagged_trainset, sents_valset, tagged_valset, sents_trainset):
    training_opt = {'feature.minfreq': 10, "num_memories": 500, "delta": 1e-8, 'linesearch': 'StrongBacktracking',
                    'c1': 0.01, 'c2': 0.01, 'max_iterations': 50, 'feature.possible_transitions': 5,
                    'feature.possible_states': 5,
                    }

    # EXISTING PARAMS
    # feature.minfreq: 10.000000
    # feature.possible_states: 0
    # feature.possible_transitions: [0, 1, 5, 10]
    # c1: [0.000000, 0.01, 0.1, 0.2, 200]
    # c2: [1.000000, 0.01, 0.1, 0.2]
    # num_memories: 6
    # max_iterations: [2147483647, 20, 10, 50, 100]
    # epsilon: 0.000010
    # delta: [0.000010, 0.1] # can range from 0.00 to 1.00
    # linesearch: ['MoreThuente', 'Backtracking', 'StrongBacktracking']
    # linesearch.max_iterations: 20
    tagger = crf.CRFTagger(verbose=True, training_opt=training_opt)
    tokens_train = [token for sentence in sents_trainset for token in sentence]
    tagger.train(tagged_trainset, 'model.crf.tagger')
    model_outputs = tagger.tag_sents(sents_valset)
    print(f"Predicted tagged val sentence {model_outputs[0]}")

    accuracy = tagger.evaluate(tagged_valset)
    print("Accuracy on the validation set:", accuracy)  # 0.9528898254063817
    predicted_labels = [tag for sentence in model_outputs for token, tag in sentence]

    return model_outputs, tagger, accuracy, predicted_labels
    # # Step 1. Define the param_grid
    # param_grid = {'C': [0.01, 0.1, 0.5, 1, 10, 100],
    #               'gamma': [1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001],
    #               'kernel': ['rbf', 'poly', 'linear']}
    # # Step 2. GridSearch and fit the model
    # grid = GridSearchCV(sklearn_crfsuite.CRF(), param_grid=param_grid, cv=3)
    # grid.fit(tokens, labels_trainset)
    #
    # best_params = grid.best_params_
    # print(best_params)


def get_best_parameters(X_train, y_train, y_val, is_tuned):
    tagger = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )
    tagger.fit(X_train, y_train)
    labels = list(tagger.classes_)
    y_pred = tagger.predict(X_val)
    accuracy = metrics_crf.flat_accuracy_score(y_val, y_pred)
    print(accuracy)
    return accuracy, y_pred, labels
    # accuracy = metrics.make_scorer(metrics.accuracy_score(gold_tokens_val, predicted_token_labels))
    # # rs = RandomizedSearchCV(estimator=tagger, param_distributions=training_opt,
    # #                         cv=3,
    # #                         verbose=1,
    # #                         n_jobs=-1,
    # #                         n_iter=50)
    # gs = GridSearchCV(estimator=tagger, param_grid=training_opt, cv=3)
    # try:
    #     gs.fit(X_train, y_train)
    # except AttributeError:
    #     pass
    # gs.best_estimator_


def tune(X_train, y_train):
    y_train = MultiLabelBinarizer().fit_transform(y_train)
    y_train = np.asarray(y_train)
    tagger = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )
    params_space = {"num_memories": [200, 500], "delta": [0.000010, 0.1],
                    'linesearch': ['MoreThuente', 'Backtracking', 'StrongBacktracking'],
                    'c1': [0.000000, 0.01, 0.1, 0.2], 'c2': [1.000000, 0.01, 0.1, 0.2]
                    }
    # training_opt = {'feature.minfreq': [5, 10], "num_memories": [200, 500], "delta": [0.000010, 0.1],
    #                 'linesearch': ['MoreThuente', 'Backtracking', 'StrongBacktracking'],
    #                 'c1': [0.000000, 0.01, 0.1, 0.2], 'c2': [1.000000, 0.01, 0.1, 0.2],
    #                 'max_iterations': [2147483647, 20, 10, 50, 100], 'feature.possible_transitions': [0, 1, 5, 10],
    #                 'feature.possible_states': [4, 5]
    #                 }
    accuracy_scorer = make_scorer(accuracy_score)
    # rs = RandomizedSearchCV(estimator=tagger, param_distributions=params_space,
    #                         cv=3,
    #                         verbose=1,
    #                         n_jobs=-1,
    #                         n_iter=50, scoring='accuracy', error_score='raise')
    # try:
    #     rs.fit(X_train, y_train)
    # except AttributeError:
    #     pass
    #
    # print(rs.best_params_)
    # print(accuracy_scorer)

    # instantiate a GridSearchCV object
    rs = GridSearchCV(tagger,
                      params_space,
                      cv=3,
                      verbose=1,
                      n_jobs=3,
                      scoring=accuracy_scorer,
                      return_train_score=True
                      )
    # fit
    try:
        rs.fit(X_train, y_train)
    except AttributeError:
        pass
    return rs.best_params_


def tune_hyper_manually(tagged_train_sents, tagged_val_sents, sentences_val):
    highest_accuracy = 0
    best_params = None
    results = []
    training_opt = {"delta": [0.000010, 0.1], 'linesearch': ['MoreThuente', 'Backtracking', 'StrongBacktracking'],
                    'c1': [0, 0.01, 0.1, 0.2], 'c2': [1, 0.01, 0.1, 0.2],
                    'max_iterations': [5, 20, 10, 50],
                    }
    for i in range(10):
        start = timer()
        params = {key: random.sample(value, 1)[0] for key, value in training_opt.items()}
        # print(params)
        tagger = crf.CRFTagger(training_opt=params)
        tagger.train(tagged_train_sents, 'model.crf.tagger')
        model_outputs = tagger.tag_sents(sentences_val)
        accuracy = tagger.evaluate(tagged_val_sents)
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_params = params
        end = timer()
        results_list = params, accuracy, end - start
        results.append(results_list)
    df = pd.DataFrame(results, columns = ['Parameters', 'Accuracy', 'Time'])
    print(df)
    print(best_params)
    return df, highest_accuracy, best_params


def k_fold_validation(tagger, X_val, y_val):
    # X_train_data = np.asarray(X_train_data)
    # y_train_data = np.asarray(y_train_data)
    # cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # rf = crf.CRFTagger()
    # visualizer = CVScores(rf, cv=cv, scoring='accuracy')
    # visualizer.fit(X_train_data, y_train_data)
    # visualizer.show()
    cv_results = cross_validate(tagger, X_val, y_val, cv=3, scoring=['f1_weighted', 'f1_micro', 'f1_macro'])
    for key, value in cv_results.items():
        print(value, '\t', key)


def print_state_features(tagger):
    pass
    # state_features = tagger.__getattribute__()
    # for (attr, label), weight in tagger:
    #     print("%0.6f %-8s %s" % (weight, label, attr))
    #
    # print("Top positive:")
    # print_state_features(Counter(crf.state_features_).most_common(20))
    #
    # print("\nTop negative:")
    # print_state_features(Counter(crf.state_features_).most_common()[-20:])


def reformat_labels(val_labels, val_predicted_labels):
    y_gold = MultiLabelBinarizer().fit_transform(val_labels)
    y_predicted = MultiLabelBinarizer().fit_transform(val_predicted_labels)
    return y_gold, y_predicted


def error_analysis(model_outputs, tagged_val_sentences):
    top_errors = {'nonstop': [], 'that': [], 'which': [], 'is': [], 'to': []}
    count = 0
    incorrectly_labeled_tokens = 0
    tokens = {}
    for i in range(len(model_outputs)):
        sentence = model_outputs[i]
        gold_sentence = tagged_val_sentences[i]
        for j in range(len(sentence)):
            token, predicted_label = sentence[j]
            real_label = gold_sentence[j][1]
            if predicted_label != real_label:
                incorrectly_labeled_tokens += 1
                count += 1
                try:
                    tokens[token] += 1
                except KeyError:
                    tokens[token] = 1
                # get the original raw text with the corresponding index for the mislabed token
                # Print only the top mislabeled
                if token in top_errors:
                    top_errors[token].append(tagged_to_sentence(gold_sentence))
                    print(f"{count} Incorrect predicted label '{predicted_label}' at token '{j, token}' in sentence: \n    {i, gold_sentence}\n")
    print(f"{incorrectly_labeled_tokens} incorrectly labeled tokens, {len(tokens.items())} of which are unique.")
    # ranked_error_tokens = sorted(tokens, key=tokens.get, reverse=True)
    ranked_error_tokens = sorted(tokens.items(), key=lambda x: x[1], reverse=True)
    print(ranked_error_tokens)
    # print(top_errors)


def tagged_to_sentence(tagged_sentence):
    tokens = [token for token, tag in tagged_sentence]
    sentence = ' '.join(tokens)
    return sentence



# 1: EXTRACT AND REFORMAT DATA
train_data, val_data, test_data = read_datasets('UD_English-Atis-master/', 'en_atis-ud-train.conllu',
                                                'en_atis-ud-dev.conllu', 'en_atis-ud-test.conllu')
# print(train_data[0])
# print(train_data[0][0])
tagged_sents_train, tagged_sents_val, tagged_sents_test = tag_datasets(train_data, val_data, test_data)
# print(f"Tagged sentences train {tagged_sents_train[0]}")
sentences_train, sentences_val, sentences_test = get_sentences_from_datasets(train_data, val_data, test_data)
tokens_train, tokens_val, tokens_test = get_tokens_from_sentences(sentences_train, sentences_val, sentences_test)
gold_sent_labels_train, gold_sent_labels_val, gold_sent_labels_test = get_sentence_gold_labels_from_datasets(tagged_sents_train,
                                                                                                             tagged_sents_val,
                                                                                                             tagged_sents_test)
gold_tokens_train, gold_tokens_val, gold_tokens_test = get_token_gold_labels(gold_sent_labels_train, gold_sent_labels_val, gold_sent_labels_test)
# print(sent2features(tagged_sents_train[0])[0])
X_train = [sent2features(s) for s in tagged_sents_train]
y_train = [sent2labels(s) for s in tagged_sents_train]
X_val = [sent2features(s) for s in tagged_sents_val]
y_val = [sent2labels(s) for s in tagged_sents_val]
# print(X_train[0])
# print(y_train[0])
# 2: TRAIN, EVALUATE, TUNE

# Baselines: basic and advanced
# accuracy_most_frequent = train_baseline(tokens_train, gold_tokens_train, sentences_val, gold_tokens_val, 'most_frequent')  # 0.23344370860927152
# accuracy_mlp = train_baseline(tokens_train, gold_tokens_train, sentences_val, gold_tokens_val, 'mlp')  # Accuracy on the val set with strategy mlp: 0.9262492474413004

# Train & evaluate model
model_outputs, tagger, accuracy, predicted_token_labels = train_model(tagged_sents_train, sentences_val,
                                                                      tagged_sents_val)  # Accuracy on the validation set: 0.9793798916315473
print(metrics.classification_report(gold_tokens_val, predicted_token_labels, zero_division=0))

# Hyperparameter tuning
# model_outputs, tagger, accuracy, predicted_token_labels = tune_hyperparameters(tagged_sents_train, sentences_val, tagged_sents_val)  # Accuracy on the validation set: 0.9759181216134859 'feature.minfreq' : 10
# model_outputs, tagger, accuracy, predicted_token_labels = tune_hyperparameters(tagged_sents_train, sentences_val, tagged_sents_val)  # Accuracy on the validation set: 0.9759181216134859 {'feature.minfreq': 10, "num_memories": 500, "delta": 1e-8}
# model_outputs, tagger, accuracy, predicted_token_labels = tune_hyperparameters(tagged_sents_train, sentences_val, tagged_sents_val)  # Accuracy on the validation set: 0.9775737507525587 {'feature.minfreq': 10, "num_memories": 500, "delta": 1e-8, 'linesearch': 'StrongBacktracking',
# 'c1': 0.1, 'c2': 0.1, 'max_iterations': 20}
# model_outputs, tagger, accuracy, predicted_token_labels = tune_hyperparameters(tagged_sents_train, sentences_val, tagged_sents_val)  # Accuracy on the validation set: 0.9798314268512944 {'feature.minfreq': 10, "num_memories": 500, "delta": 1e-8, 'linesearch': 'StrongBacktracking',
# 'c1': 0.1, 'c2': 0.1, 'max_iterations': 20}
# model_outputs, tagger, accuracy, predicted_token_labels = tune_hyperparameters(tagged_sents_train, sentences_val, tagged_sents_val)  # Accuracy on the validation set: 0.9802829620710415 {'feature.minfreq': 10, "num_memories": 500, "delta": 1e-8, 'linesearch': 'StrongBacktracking',
#                     # 'c1': 0.01, 'c2': 0.01, 'max_iterations': 50, 'feature.possible_transitions': True}
# model_outputs, tagger, accuracy, predicted_token_labels = tune_hyperparameters(tagged_sents_train, sentences_val, tagged_sents_val)  # Accuracy on the validation set: 0.7987658037326911 {'feature.minfreq': 10, "num_memories": 500, "delta": 1e-8, 'linesearch': 'StrongBacktracking',
# 'c1': 200, 'c2': 0.01, 'max_iterations': 50, 'feature.possible_transitions': True}
model_outputs, tagger, accuracy, predicted_token_labels = tune_hyperparameters(tagged_sents_train,
                                                                                                sentences_val, tagged_sents_val,
                                                                                                sentences_train)  # Accuracy on the validation set:
ConfusionMatrixDisplay.from_predictions(gold_tokens_val, predicted_token_labels, xticks_rotation='vertical')
plt.grid(None)
plt.show()
# accuracy, y_pred, labels = get_best_parameters(X_train, y_train, y_val, is_tuned=False) # 0.9996989765201686
# best_params = tune(X_train, y_train)
dataframe, best_accuracy, best_parameters = tune_hyper_manually(tagged_sents_train, tagged_sents_val, gold_sent_labels_val)
# k_fold_validation(tagger, X_val, y_val)
error_analysis(model_outputs, tagged_sents_val)
# print(tagged_sents_val)




##########################

# Now try with a dataset in Spanish, that is much larger

# train_data, val_data, test_data = read_datasets('UD_Spanish-AnCora-master/', 'es_ancora-ud-train.conllu', 'es_ancora-ud-dev.conllu', 'es_ancora-ud-test.conllu')

