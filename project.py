# Sandra Sanchez
# NLP Eval project
import pickle
import random
from collections import Counter

from timeit import default_timer as timer
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate, permutation_test_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn_crfsuite import metrics as metrics_crf
from tqdm import tqdm
from yellowbrick.model_selection import CVScores
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

SEED = 10


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

    print(f"There are {len(tokens_train)} words/tokens in the training set.")
    print(f"There are {len(tokens_val)} words/tokens in the validating set.")
    print(f"There are {len(tokens_test)} words/tokens in the test set.")
    return tokens_train, tokens_val, tokens_test


def get_token_gold_labels(gold_sent_labels_train, gold_sent_labels_val, gold_sent_labels_test):
    gold_tokens_train = [label for sentence in gold_sent_labels_train for label in sentence]
    gold_tokens_val = [label for sentence in gold_sent_labels_val for label in sentence]
    gold_tokens_test = [label for sentence in gold_sent_labels_test for label in sentence]
    return gold_tokens_train, gold_tokens_val, gold_tokens_test


def train_baseline(tokens_train, gold_train, tokens_val, gold_val, strategy):
    if strategy == 'most_frequent':
        classifier = DummyClassifier(strategy='most_frequent')
        classifier.fit(tokens_train, gold_train)
        multiclass_predictions = classifier.predict(tokens_val)
        multiclass_imbalanc_probs = classifier.predict_proba(tokens_val)
        accuracy = classifier.score(tokens_val, gold_val)
        print(f'Accuracy on the val set with baseline A strategy {strategy}: {accuracy}')
    elif strategy == 'mlp':
        classifier = MLPClassifier(verbose=True)
        X_train, y_train, X_val = transform_raw_data_into_matrix(tokens_train, gold_train, tokens_val)
        classifier.fit(X_train, y_train)
        multiclass_predictions = classifier.predict(X_val)
        multiclass_imbalanc_probs = classifier.predict_proba(X_val)
        accuracy = classifier.score(X_val, gold_val)
        print(f'Accuracy on the val set with baseline B strategy {strategy}: {accuracy}')

    return accuracy


def transform_raw_data_into_matrix(train_tokens, train_labels, dev_tokens):
    assert len(train_tokens) == len(train_labels)
    # print(sentences)
    y = np.array(train_labels)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_tokens)
    X_dev = vectorizer.transform(dev_tokens)
    return X_train, y, X_dev


def train_model(tagged_train_sents, sentences_val, tagged_val_sents, model):
    print('Training model...')
    if model == 'crf':
        type = 'A'
        tagger = crf.CRFTagger(feature_func=sent2features)
        tagger.train(tagged_train_sents, 'model.crf.tagger')
    elif model == 'crf_es':
        type = 'A'
        tagger = crf.CRFTagger()
        tagger.train(tagged_train_sents, 'model.crf_es.tagger')
    elif model == 'hmm':
        type = 'B'
        trainer = hmm.HiddenMarkovModelTrainer()
        tagger = trainer.train_supervised(tagged_train_sents)
    elif model == 'hmm_es':
        type = 'B'
        trainer = hmm.HiddenMarkovModelTrainer()
        tagger = trainer.train_supervised(tagged_train_sents)
    model_outputs = tagger.tag_sents(sentences_val)
    print(f"This is an example of a sentence tagged by the model {model}: \n {model_outputs[0]}")
    accuracy = tagger.evaluate(tagged_val_sents)
    print(f"Accuracy on the validation set with {model}, model {type} : {accuracy}")
    predicted_labels = [tag for sentence in model_outputs for token, tag in sentence]

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


def train_crf_suite(X_train, X_val, y_train, y_val):
    tagger = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )
    tagger.fit(X_train, y_train)
    labels = list(tagger.classes_)
    y_pred = tagger.predict(X_val)
    accuracy = metrics_crf.flat_accuracy_score(y_val, y_pred)
    print(f"Accuracy on the validation set with the crf_suite (model B): {accuracy}")
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


def tune_hyper_manually(tagged_train_sents, tagged_val_sents, sentences_val, set):
    print('Performing hyperparameter tuning...')
    highest_accuracy = 0
    best_params = None
    results = []
    training_opt = {"delta": [0.000010, 0.001, 0.1], 'linesearch': ['MoreThuente', 'Backtracking', 'StrongBacktracking'],
                    'c1': [0, 0.01, 0.1, 0.2], 'c2': [1, 0.01, 0.1, 0.2],
                    'max_iterations': [20, 10, 50, 100], 'feature.possible_transitions': [0, 1, 5, 10]
                    }
    for i in tqdm(range(20)):
        start = timer()
        params = {key: random.sample(value, 1)[0] for key, value in training_opt.items()}
        # print(params)
        tagger = crf.CRFTagger(training_opt=params)
        if set == 'en':
            tagger.train(tagged_train_sents, 'model.crf_tuned.tagger')
        elif set == 'spa':
            tagger.train(tagged_train_sents, 'model.crf_es_tuned.tagger')
        model_outputs = tagger.tag_sents(sentences_val)
        accuracy = tagger.evaluate(tagged_val_sents)
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_params = params
        end = timer()
        results_list = params, accuracy, end - start
        results.append(results_list)
    df = pd.DataFrame(results, columns = ['Parameters', 'Accuracy', 'Time'])
    if set == 'en':
        df.to_csv('results_tuning_en.csv')
        save_data(best_params, 'best_params_en.pkl')
    elif set == 'spa':
        df.to_csv('results_tuning_spa.csv')
        save_data(best_params, 'best_params_spa.pkl')
    print(df)
    print(best_params)
    print(f"Accuracy on the validation set with the best parameters for CRF (model A): {highest_accuracy}")

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
    top_errors = {'nonstop': [], 'that': [], 'which': [], 'is': [], 'to': [],
                  'que': [], 'como': [], 'la': [], 'cuando': [], 'mientras': []}  #todo: defaultdict?
    count = 0
    incorrectly_labeled_token_occurrences = 0
    tokens = {}
    for i in range(len(model_outputs)):
        sentence = model_outputs[i]
        gold_sentence = tagged_val_sentences[i]
        for j in range(len(sentence)):
            token, predicted_label = sentence[j]
            real_label = gold_sentence[j][1]
            if predicted_label != real_label:
                incorrectly_labeled_token_occurrences += 1
                count += 1
                try:
                    tokens[token] += 1
                except KeyError:
                    tokens[token] = 1
                # get the original raw text with the corresponding index for the mislabed token
                # Print only the top mislabeled
                if token in top_errors:
                    top_errors[token].append(tagged_to_sentence(gold_sentence))
                    print(f"{count} Incorrect predicted label '{predicted_label}' at token '{j, token}' in sentence {i}: \n    {tagged_to_sentence(gold_sentence)}\n")
    print(f"{incorrectly_labeled_token_occurrences} incorrectly labeled tokens, {len(tokens.items())} of which are unique.")
    ranked_error_tokens = sorted(tokens.items(), key=lambda x: x[1], reverse=True)
    print()
    print('These are the top 5 mislabeled tokens:')
    for token, occurrence in ranked_error_tokens[:5]:
        print(f"'{token}': {occurrence} times.")


def tagged_to_sentence(tagged_sentence):
    tokens = [token for token, tag in tagged_sentence]
    sentence = ' '.join(tokens)
    return sentence


def get_previous_n_posterior_word(sentence, token):
    for i in range(len(sentence)):
        word = sentence[i]
        if word[0] == token:
            print(sentence[i - 1], word, sentence[i + 1])


def check_token_labeling(tagged_train, tagged_val, token):
    print()
    print()
    print('___________________These are the validating-set sentences._____________________')
    print()
    print()
    for sentence in tagged_val:
        get_previous_n_posterior_word(sentence, token)

    print()
    print()
    print('___________________These are the training-set sentences._______________________')
    print()
    print()
    for sentence in tagged_train:
        get_previous_n_posterior_word(sentence, token)


def t_statistic(eval_metric_a: float, eval_metric_b: float) -> float:
    return eval_metric_a - eval_metric_b


def paired_randomization_test(outputs_a, outputs_b, gold,
                              rejection_level: float, R: int) -> (float, bool):
    assert len(outputs_a) == len(outputs_b)
    c = 0  # Set c = 0
    acc_a = accuracy_score(gold, outputs_a)
    acc_b = accuracy_score(gold, outputs_b)
    actual_statistic = t_statistic(acc_a, acc_b)  # Compute actual statistic of score differences |SX − SY| on test data
    for i in range(0, R):  # for all random shuffles r = 0,...,R do
        # for all sentences in test set do
        # Shuffle variable tuples between system X and Y with probability 0.5
        shuffled_a, shuffled_b = shuffle(outputs_a, outputs_b)
        pseudo_stat = t_statistic(accuracy_score(gold, shuffled_a), accuracy_score(gold, shuffled_b))  # Compute pseudo-statistic |SXr − SYr | on shuffled data
        if pseudo_stat >= actual_statistic:  # If |SXr − SYr | >= |SX − SY|
            c += 1
    p_value = (c + 1) / (R + 1)
    if p_value <= rejection_level:  # Reject null hypothesis if p is less than or equal to specified rejection level.
        reject = True
        print(f"The p-value is {p_value}, therefore smaller or equal than the rejection level of {rejection_level}, "
              f"so we can reject the null hypotesis.")
        print(f"The difference in  performance for models A and B is thus statistically significant for this sample.")
    else:
        reject = False
        print(f"The p-value is {p_value}, therefore greater than the rejection level of {rejection_level}, "
              f"so we fail to reject the null hypotesis.")
        print(f"The difference in  performance for models A and B is thus not statistically significant.")

    return p_value, reject


def extract_and_reformat_data(directory_name, train_setset_file, val_set_file, test_set_file):
    train_data, val_data, test_data = read_datasets(directory_name, train_setset_file,
                                                    val_set_file, test_set_file)
    tagged_sents_train, tagged_sents_val, tagged_sents_test = tag_datasets(train_data, val_data, test_data)
    sentences_train, sentences_val, sentences_test = get_sentences_from_datasets(train_data, val_data, test_data)
    if 'English-Atis' in directory_name:
        print("The dataset Atis, (English) is composed of:")
    elif 'Spanish-AnCora' in directory_name:
        print("The dataset AnCora, (Spanish) is composed of:")
    else:
        print('The dataset is composed of:')
    print(f"{len(tagged_sents_train)} sentences for the training set.")
    print(f"{len(tagged_sents_val)} sentences for the validating set.")
    print(f"{len(tagged_sents_test)} sentences for the test set.")

    return tagged_sents_train, tagged_sents_val, tagged_sents_test, sentences_train, sentences_val, sentences_test


def get_x_and_y_features(tagged_sents_train, tagged_sents_val):
    X_train = [sent2features(s) for s in tagged_sents_train]
    y_train = [sent2labels(s) for s in tagged_sents_train]
    X_val = [sent2features(s) for s in tagged_sents_val]
    y_val = [sent2labels(s) for s in tagged_sents_val]

    return X_train, y_train, X_val, y_val


def save_data(data: any, filename: any) -> None:
    """Save data into file_name (.pkl file)to save time."""
    with open(filename, mode="wb") as file:
        pickle.dump(data, file)
        print(f'Data saved in {filename}')


def load_data(filename: any) -> any:
    """Load pre-saved data."""
    with open(filename, "rb") as file:
        output = pickle.load(file)
    print(f'Loading  data  pre-saved as {filename}...')
    return output


def word2features(sent, i):
    features = []
    word = sent[i]
    print(word)

    # features = {
    #     'bias': 1.0,
    #     'word.lower()': word.lower(),
    #     'word[-3:]': word[-3:],
    #     'word[-2:]': word[-2:],
    #     'word.isupper()': word.isupper(),
    #     'word.istitle()': word.istitle(),
    #     'word.isdigit()': word.isdigit(),
    #     'postag': postag,
    #     'postag[:2]': postag[:2],
    # }
    # if i > 0:
    #     word1 = sent[i-1][0]
    #     postag1 = sent[i-1][1]
    #     features.update({
    #         '-1:word.lower()': word1.lower(),
    #         '-1:word.istitle()': word1.istitle(),
    #         '-1:word.isupper()': word1.isupper(),
    #         '-1:postag': postag1,
    #         '-1:postag[:2]': postag1[:2],
    #     })
    # else:
    #     features['BOS'] = True
    #
    # if i < len(sent)-1:
    #     word1 = sent[i+1][0]
    #     postag1 = sent[i+1][1]
    #     features.update({
    #         '+1:word.lower()': word1.lower(),
    #         '+1:word.istitle()': word1.istitle(),
    #         '+1:word.isupper()': word1.isupper(),
    #         '+1:postag': postag1,
    #         '+1:postag[:2]': postag1[:2],
    #     })
    # else:
    #     features['EOS'] = True
    word1 = sent[i-1]
    postag1 = sent[i-1]
    features.append(word.lower())
    # features.append(word.istitle())
    # features.append(word.isupper())
    features = [f.encode('utf-8') for f in features]


    return features


def sent2features(sent, index):
    return word2features(sent, index)


def sent2labels(sent):
    return [label for token, label in sent]


def sent2tokens(sent):
    return [token for token, label in sent]



# Train & evaluate model
if __name__=='__main__':
    # 1: EXTRACT AND REFORMAT DATA
    tagged_sents_train, tagged_sents_val, tagged_sents_test, sentences_train, sentences_val, sentences_test = \
        extract_and_reformat_data('UD_English-Atis-master/', 'en_atis-ud-train.conllu',
                                  'en_atis-ud-dev.conllu',
                                  'en_atis-ud-test.conllu')  # 4274 training set, 572 validating set, 586  test
    first_sents = tagged_sents_train[:2]
    print(first_sents)
    features = []
    for sent in first_sents:
        tokens, labels = zip(*sent)
        feat = [sent2features(tokens, i) for i in range(len(tokens))]
        features.append(feat)
    print(features)
    # tokens_train, tokens_val, tokens_test = get_tokens_from_sentences(
    #     sentences_train, sentences_val, sentences_test)  # 48655, 6644, 6580
    # gold_sent_labels_train, gold_sent_labels_val, gold_sent_labels_test = get_sentence_gold_labels_from_datasets(
    #     tagged_sents_train, tagged_sents_val, tagged_sents_test)
    # gold_tokens_train, gold_tokens_val, gold_tokens_test = get_token_gold_labels(
    #     gold_sent_labels_train, gold_sent_labels_val, gold_sent_labels_test)

    # X_train, y_train, X_val, y_val = get_x_and_y_features(tagged_sents_train, tagged_sents_val)

    # 2: TRAIN, EVALUATE, TUNE

    # Baselines: basic and advanced
    # accuracy_most_frequent = train_baseline(tokens_train, gold_tokens_train, tokens_val, gold_tokens_val, 'most_frequent')  # 0.23344370860927152
    # accuracy_mlp = train_baseline(tokens_train, gold_tokens_train, tokens_val, gold_tokens_val, 'mlp')  # Accuracy on the val set with strategy mlp: 0.9262492474413004
    model_outputs, tagger, accuracy, predicted_token_labels = train_model(tagged_sents_train, sentences_val,
                                                                          tagged_sents_val, 'crf')  # Accuracy on the validation set with crf.CRFTagger default: 0.9793798916315473
    # print(metrics.classification_report(gold_tokens_val, predicted_token_labels, zero_division=0))
    # ConfusionMatrixDisplay.from_predictions(gold_tokens_val, predicted_token_labels, xticks_rotation='vertical')
    # plt.grid(None)
    # plt.savefig("confusion_matrix.png")
    # plt.show()

# Hyperparameter tuning
# best_params = tune(X_train, y_train) NOOOOOO
# dataframe, best_accuracy, best_parameters = tune_hyper_manually(tagged_sents_train, tagged_sents_val, gold_sent_labels_val)  # 0.9850993377483444
# k_fold_validation(tagger, X_val, y_val) NOOOOOOO

# Linguistic Error analysis
# error_analysis(model_outputs, tagged_sents_val)
# check_token_labeling(tagged_sents_train, tagged_sents_val, 'which')
# check_token_labeling(tagged_sents_train, tagged_sents_val, 'that')

# Performance evaluation: statistical significance testing
# accuracy_model_b, pred_b, labels = train_crf_suite(X_train, X_val, y_train, y_val)  # Nooooo
# model_outputs_b, tagger_b, accuracy_b, predicted_token_labels_b = train_model(tagged_sents_train, sentences_val,
#                                                                       tagged_sents_val, 'hmm')  # 0.9528898254063817
# test_statistic_measure = test_statistic(best_accuracy, accuracy_model_b)
# paired_randomization_test(predicted_token_labels, predicted_token_labels_b, gold_tokens_val, rejection_level= 0.05, R=1000)  # Reject = True

##########################

# Now try with a dataset in Spanish, that is much larger

# 1: EXTRACT AND REFORMAT DATA

# tagged_sents_train_s2, tagged_sents_val_s2, tagged_sents_test_s2, sentences_train_s2, sentences_val_s2, sentences_test_s2 = extract_and_reformat_data(
#     'UD_Spanish-AnCora-master/', 'es_ancora-ud-train.conllu', 'es_ancora-ud-dev.conllu', 'es_ancora-ud-test.conllu')  # 14287 training sents, 1654 val, 1721 test
# tokens_train_s2, tokens_val_s2, tokens_test_s2 = get_tokens_from_sentences(
#     sentences_train_s2, sentences_val_s2, sentences_test_s2)  # 469366, 55482, 55603
# gold_sent_labels_train_s2, gold_sent_labels_val_s2, gold_sent_labels_test_s2 = get_sentence_gold_labels_from_datasets(
#     tagged_sents_train_s2, tagged_sents_val_s2, tagged_sents_test_s2)
# gold_tokens_train_s2, gold_tokens_val_s2, gold_tokens_test_s2 = get_token_gold_labels(
#     gold_sent_labels_train_s2, gold_sent_labels_val_s2, gold_sent_labels_test_s2)

# 2: TRAIN, EVALUATE, TUNE

# Baselines: basic and advanced
# accuracy_most_frequent = train_baseline(tokens_train_s2, gold_tokens_train_s2, tokens_val_s2, gold_tokens_val_s2, 'most_frequent')  # 0.17295699506146137
# accuracy_mlp = train_baseline(tokens_train_s2, gold_tokens_train_s2, tokens_val_s2, gold_tokens_val_s2, 'mlp')  # Accuracy on the val set with baseline B strategy mlp: 0.8396957571825097

# # Train & evaluate model
# model_outputs_s2, tagger_s2, accuracy_s2, predicted_token_labels_s2 = train_model(
#     tagged_sents_train_s2, sentences_val_s2, tagged_sents_val_s2, 'crf_es')  # Accuracy on the validation set with crf.CRFTagger: 0.9697018852961321
# print(metrics.classification_report(gold_tokens_val_s2, predicted_token_labels_s2, zero_division=0))
# save_data(predicted_token_labels_s2, 'predicted_token_labels_s2.pkl')
# save_data(model_outputs_s2, 'model_outputs_s2.pkl')

# ConfusionMatrixDisplay.from_predictions(gold_tokens_val_s2, predicted_token_labels_s2, xticks_rotation='vertical')
# plt.grid(None)
# plt.savefig("confusion_matrix_spa.png")
# plt.show()

# dataframe, best_accuracy, best_parameters = tune_hyper_manually(
#     tagged_sents_train_s2, tagged_sents_val_s2, gold_sent_labels_val_s2, 'spa')  # Accuracy on the validation set with the best parameters for CRF (model A): 0.9771457409610325

# Linguistic Error analysis
# error_analysis(model_outputs_s2, tagged_sents_val_s2)
## 1681 incorrectly labeled tokens, 1135 of which are unique.
## Top mislabeled: que, como, la, cuando, mientras
# check_token_labeling(tagged_sents_train_s2, tagged_sents_val_s2, 'que')
# check_token_labeling(tagged_sents_train_s2, tagged_sents_val_s2, 'como')

# # Performance evaluation: statistical significance testing
# model_outputs_b_s2, tagger_b_s2, accuracy_b_s2, predicted_token_labels_b_s2 = train_model(
#     tagged_sents_train_s2, sentences_val_s2, tagged_sents_val_s2, 'hmm')  # Accuracy on the validation set with hmm: 0.5158609999639523
# paired_randomization_test(
#     predicted_token_labels_s2, predicted_token_labels_b_s2, gold_tokens_val_s2, rejection_level= 0.05, R=1000)  # Reject = True
