# Sandra Sanchez
# NLP Evaluation Systems  Project
# 31.03.2023

import pickle
import random
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import sklearn_crfsuite
from conllu import parse
from matplotlib import pyplot as plt
from nltk.tag import hmm, crf
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn_crfsuite import metrics as metrics_crf
from tqdm import tqdm

SEED = 10


def read_datasets(corpus_dirname, train_filename, dev_filename, test_filename):
    """Find path for corpus directory, read and return training, development and test sets."""

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
    """Return POS-tagged sentences from dataset splits."""

    tagged_sents_train = [[(word['form'], word['upos']) for word in sent] for sent in train_set]
    tagged_sents_val = [[(word['form'], word['upos']) for word in sent] for sent in val_set]
    tagged_sents_test = [[(word['form'], word['upos']) for word in sent] for sent in test_set]

    return tagged_sents_train, tagged_sents_val, tagged_sents_test


def get_sentences_from_datasets(train_set, val_set, test_set):
    """Return sentences from dataset splits as lists of strings."""

    sents_train = [[word['form'] for word in sent] for sent in train_set]
    sents_val = [[word['form'] for word in sent] for sent in val_set]
    sents_test = [[word['form'] for word in sent] for sent in test_set]

    return sents_train, sents_val, sents_test


def get_sentence_gold_labels_from_datasets(tagged_train, tagged_val, tagged_test):
    """Return POS-tag labels for every token in every sentence."""

    gold_sent_labels_train = [[(word[1]) for word in sent] for sent in tagged_train]
    gold_sent_labels_val = [[(word[1]) for word in sent] for sent in tagged_val]
    gold_sent_labels_test = [[(word[1]) for word in sent] for sent in tagged_test]

    return gold_sent_labels_train, gold_sent_labels_val, gold_sent_labels_test


def get_tokens_from_sentences(sent_train, sent_val, sent_test):
    """Return list of tokens (str) for every sentence in data split."""

    tokens_train = [token for sentence in sent_train for token in sentence]
    tokens_val = [token for sentence in sent_val for token in sentence]
    tokens_test = [token for sentence in sent_test for token in sentence]

    print(f"There are {len(tokens_train)} words/tokens in the training set.")
    print(f"There are {len(tokens_val)} words/tokens in the validating set.")
    print(f"There are {len(tokens_test)} words/tokens in the test set.")
    return tokens_train, tokens_val, tokens_test


def get_token_gold_labels(gold_sent_labels_train, gold_sent_labels_val, gold_sent_labels_test):
    """Return list of POS-tags for all tokens in a set split."""

    gold_tokens_train = [label for sentence in gold_sent_labels_train for label in sentence]
    gold_tokens_val = [label for sentence in gold_sent_labels_val for label in sentence]
    gold_tokens_test = [label for sentence in gold_sent_labels_test for label in sentence]
    return gold_tokens_train, gold_tokens_val, gold_tokens_test


def train_baseline(tokens_train, gold_train, tokens_val, gold_val, strategy):
    """
    Train a baseline depending on specified strategy.
    Fit model, predict labels for validating set, compute
    and return accuracy for the validating set.
    """

    print('Training baseline...')
    if strategy == 'most_frequent':
        classifier = DummyClassifier(strategy='most_frequent', random_state=SEED)
        classifier.fit(tokens_train, gold_train)
        multiclass_predictions = classifier.predict(tokens_val)
        multiclass_imbalanc_probs = classifier.predict_proba(tokens_val)
        accuracy = classifier.score(tokens_val, gold_val)
        print(f'Accuracy on the val set with baseline A strategy {strategy}: {accuracy}')
    elif strategy == 'mlp':
        classifier = MLPClassifier(verbose=True, random_state=SEED)
        X_train, y_train, X_val = transform_raw_data_into_matrix(tokens_train, gold_train, tokens_val)
        classifier.fit(X_train, y_train)
        multiclass_predictions = classifier.predict(X_val)
        multiclass_imbalanc_probs = classifier.predict_proba(X_val)
        accuracy = classifier.score(X_val, gold_val)
        print(f'Accuracy on the val set with baseline B strategy {strategy}: {accuracy}')

    return accuracy


def transform_raw_data_into_matrix(train_tokens, train_labels, dev_tokens):
    """
    Transform list of tokens and labels into two arrays
    in order to prepare data for training a model.
    """
    assert len(train_tokens) == len(train_labels)
    y = np.array(train_labels)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_tokens)
    X_dev = vectorizer.transform(dev_tokens)
    return X_train, y, X_dev


def train_model(tagged_train_sents, sentences_val, tagged_val_sents, model):
    """
    Choose paradigm and implementation, train/save model. Use model to
    tag sentences from validation set and calculate accuracy accordingly.
    :param tagged_train_sents: list of lists of tuples with token, POS-tag values
        used for supervised training.
    :param sentences_val: List of lists of tokens(str) that the model will tag
    :param tagged_val_sents: list of lists of tuples with token, POS-tag values
        used for calculating accuracy of the model.
    :param model: type of paradigm used for training, indicates implementation type
    :return: outputs, tagger, accuracy score, predicted_labels
    """
    print('Training model...')
    if model == 'crf':
        type = 'A'
        tagger = crf.CRFTagger()
        tagger.train(tagged_train_sents, 'model.crf.tagger')
    elif model == 'crf_es':
        type = 'A'
        tagger = crf.CRFTagger()
        tagger.train(tagged_train_sents, 'model.crf_es.tagger')
    elif model == 'crf_func':
        type = 'A'
        tagger = crf.CRFTagger(feature_func=sent2features)
        tagger.train(tagged_train_sents, 'model.crf_func.tagger')
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
    print(f"Accuracy on the validation set with {model}, paradigm {type} : {accuracy}")
    predicted_labels = [tag for sentence in model_outputs for token, tag in sentence]

    return model_outputs, tagger, accuracy, predicted_labels


def train_crf_sklearn(X_train, X_val, y_train, y_val):
    """
    Train model on crf paradigm with sklearn implementation.
    Iterate over all possible estimating algorithms to train model.
    :param X_train: training sentences as numpy array
    :param X_val: validating sentences as numpy array
    :param y_train: training labels as numpy array
    :param y_val: validating labels as numpy array
    :return: highest_accuracy, y_preds, all_labels
    """
    algorithms = ['lbfgs', 'l2sgd', 'ap', 'pa', 'arow']
    highest_accuracy = 0
    y_preds = None
    all_labels = None
    best_alg = None
    for a in algorithms:
        tagger = sklearn_crfsuite.CRF(
            algorithm=a,
            max_iterations=100,
            all_possible_transitions=True
        )
        tagger.fit(X_train, y_train)
        labels = list(tagger.classes_)
        y_pred = tagger.predict(X_val)
        accuracy = metrics_crf.flat_accuracy_score(y_val, y_pred)
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            y_preds = y_pred
            all_labels = labels
            best_alg = a
        print(f"Accuracy on the validation set with the crf_suite (model A), algorithm {a}: {accuracy}")
    print(f"The highest accuracy ({highest_accuracy}) was achieved with the algorithm '{best_alg}'")
    return highest_accuracy, y_preds, all_labels


def tune_training_opt(tagged_train_sents, tagged_val_sents, sentences_val, set):
    """
    Try out random different training options for training a crf model.
    Return parameter combination that achieves highest accuracy
    and print out dataframe with used combinations.
    :param tagged_train_sents: list of lists of tuples with token, POS-tag values
        used for supervised training.
    :param tagged_val_sents: list of lists of tuples with token, POS-tag values
        used for calculating accuracy of the model.
    :param sentences_val: List of lists of tokens(str) that the model will tag
    :param set: specifies dataset: 'en' = English, 'spa' = Spanish
    :return: df, highest_accuracy, best_params
    """
    print('Performing hyperparameter tuning...')
    highest_accuracy = 0
    best_params = None
    results = []
    training_opt = {"delta": [0.000010, 0.001, 0.1],
                    'linesearch': ['MoreThuente', 'Backtracking', 'StrongBacktracking'],
                    'c1': [0, 0.01, 0.1, 0.2], 'c2': [1, 0.01, 0.1, 0.2],
                    'max_iterations': [20, 10, 50, 100], 'feature.possible_transitions': [0, 1, 5, 10]
                    }
    for i in tqdm(range(20)):
        start = timer()
        params = {key: random.sample(value, 1)[0] for key, value in training_opt.items()}
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
    df = pd.DataFrame(results, columns=['Parameters', 'Accuracy', 'Time'])
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


def train_best_params_n_function(tagged_train_sents, computed_best_params, sentences_val, tagged_val_sents, model):
    """
    Use best training options from tune_training_opt and sent2features
    function to do fine tuning of model.
    :param tagged_train_sents: list of lists of tuples with token, POS-tag values
        used for supervised training.
    :param computed_best_params: dict with best combination of training parameters.
    :param sentences_val: List of lists of tokens(str) that the model will tag
    :param tagged_val_sents: list of lists of tuples with token, POS-tag values
        used for calculating accuracy of the model.
    :param model: type of paradigm used for training, indicates implementation type
    :return: model_outputs, tagger, accuracy, predicted_labels.
    """
    print('Training model...')
    type = 'A'
    tagger = crf.CRFTagger(feature_func=sent2features, training_opt=computed_best_params)
    tagger.train(tagged_train_sents, 'model.crf_tuned.tagger')
    model_outputs = tagger.tag_sents(sentences_val)
    print(f"This is an example of a sentence tagged by the model {model}: \n {model_outputs[0]}")
    accuracy = tagger.evaluate(tagged_val_sents)
    print(f"Accuracy on the validation set with {model}, paradigm {type} : {accuracy}")
    predicted_labels = [tag for sentence in model_outputs for token, tag in sentence]

    return model_outputs, tagger, accuracy, predicted_labels


def perform_error_analysis(model_outputs, tagged_val_sentences):
    """
    Compare model-generated outputs against gold standard and print out
    mislabeled tokens with their context sentence and top 5 errors.
    :param model_outputs: list of lists of tuples(token, POS-tag) with
        model predictions
    :param tagged_val_sentences: gold standard
    """
    top_errors = {'nonstop': [], 'that': [], 'which': [], 'is': [], 'to': [],
                  'que': [], 'como': [], 'la': [], 'cuando': [], 'mientras': [],
                  'much': []}
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
                if token in top_errors:
                    top_errors[token].append(transform_tagged_sent_to_sentence(gold_sentence))
                    # get the original raw text with the corresponding index for the mislabed token
                    print(
                        f"{count} Incorrect predicted label '{predicted_label}' at token "
                        f"'{j, token}' in sentence {i}: \n    {transform_tagged_sent_to_sentence(gold_sentence)}\n")
    print(
        f"{incorrectly_labeled_token_occurrences} incorrectly labeled tokens, {len(tokens.items())} of which are unique.")
    ranked_error_tokens = sorted(tokens.items(), key=lambda x: x[1], reverse=True)
    print()
    print('These are the top 5 mislabeled tokens:')
    # Print only the top mislabeled
    for token, occurrence in ranked_error_tokens[:5]:
        print(f"'{token}': {occurrence} times.")


def transform_tagged_sent_to_sentence(tagged_sentence):
    """Take list of tagged tokens and return str with all tokens."""
    tokens = [token for token, tag in tagged_sentence]
    sentence = ' '.join(tokens)
    return sentence


def get_previous_n_posterior_word(sentence, token):
    """
    Find previous and posterior tokens to a word in a tagged sentence
    and print out the three elements with their POS-tags.
    """
    for i in range(len(sentence)):
        word = sentence[i]
        if word[0] == token:
            if i > 0:
                previous_word = sentence[i - 1]
            else:
                previous_word = ''
            if i == len(sentence) - 1:
                next_word = ''
            else:
                next_word = sentence[i + 1]
            print(previous_word, word, next_word)


def get_token_previous_posterior_with_labels(tagged_train, tagged_val, token):
    """Call get_previous_n_posterior_word for sentences containing a certain token."""
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


def t_statistic(eval_metric_a: float, eval_metric_b: float):
    """Compute and return test statistic on two evaluation metrics."""
    return eval_metric_a - eval_metric_b


def compute_test_statistic(outputs_a, outputs_b, gold):
    """Calculate accuracy for models A and B and call t_statistic."""
    acc_a = accuracy_score(outputs_a, gold)
    acc_b = accuracy_score(outputs_b, gold)
    return t_statistic(acc_a, acc_b)


def paired_randomization_test(outputs_a, outputs_b, gold,
                              rejection_level: float, R: int) -> (float, bool):
    """
    Perform two-tailed randomization test to compute statistical significance.
    Return p-value and reject or fail to reject null hypothesis accordingly.
    :param outputs_a: Tags predicted by model A on gold set
    :param outputs_b: Tags predicted by model B on gold set
    :param gold: Real labels for validation set
    :param rejection_level: threshold to reject or fail to reject null hypothesis
    :param R: number of repetitions
    :return: p-value, reject
    """
    assert len(outputs_a) == len(outputs_b)
    test_stat = compute_test_statistic(outputs_a, outputs_b, gold)
    results = []

    for i in range(0, R):
        swapped_outputs_a = []
        swapped_outputs_b = []
        for sent_index in range(len(gold)):
            swap = np.random.uniform(0, 1)
            swapped_outputs_a, swapped_outputs_b = update_outputs(swapped_outputs_a, swapped_outputs_b,
                                                                  outputs_a[sent_index], outputs_b[sent_index],
                                                                  swap < 0.5)

        pseudo_stat = compute_test_statistic(np.array(swapped_outputs_a), np.array(swapped_outputs_b), gold)
        results.append(int(abs(pseudo_stat) >= abs(test_stat)))

    p_value = (sum(results) + 1) / (len(results) + 1)

    if p_value <= rejection_level:  # Reject null hypothesis if p is less than or equal to specified rejection level.
        reject = True
        print(f"The p-value is {p_value}, therefore smaller or equal than the rejection level of {rejection_level}, "
              f"so we can reject the null hypotesis.")
        print(f"The difference in  performance for models A and B is thus statistically significant.")
    else:
        reject = False
        print(f"The p-value is {p_value}, therefore greater than the rejection level of {rejection_level}, "
              f"so we fail to reject the null hypotesis.")
        print(f"The difference in  performance for models A and B is thus not statistically significant.")

    return p_value, reject


def extract_and_reformat_data(directory_name, train_setset_file, val_set_file, test_set_file):
    """
    Call read_datasets, tag_datasets and get_sentences_from_datasets on dataset splits
    and return reformatted data for model training and evaluating.
    :param directory_name: directory where datasets splits are
    :param train_setset_file: file with training set
    :param val_set_file: file with validating set
    :param test_set_file: file with test set
    :return: tagged_sents_train, tagged_sents_val, tagged_sents_test,
        sentences_train, sentences_val, sentences_test
    """
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


def get_X_and_y_features_from_datasets(tagged_sents_train, tagged_sents_val):
    """Take tagged sents and return features and according labels."""
    X_train = []
    y_train = []
    for sent in tagged_sents_train:
        tokens, labels = zip(*sent)
        features = [sent2features(tokens, i) for i in range(len(tokens))]
        X_train.append(features)
        labels = [label for label in labels]
        y_train.append(labels)

    X_val = []
    y_val = []
    for sent in tagged_sents_val:
        tokens, labels = zip(*sent)
        features = [sent2features(tokens, i) for i in range(len(tokens))]
        X_val.append(features)
        labels = [label for label in labels]
        y_val.append(labels)

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
    """Return list of features for every token in a sentence."""
    features = []
    word = sent[i]

    features.append(word.lower())
    features = [f.encode('utf-8') for f in features]
    features.append(word[-3:])
    features.append(word[-2:])
    features.append(str(word.istitle()))
    features.append(str(word.isupper()))
    features.append(str(word.istitle()))
    features.append(str(word.isdigit()))

    if i > 0:
        features.append(sent[i - 1])
    else:
        features.append(str('BOS'))
    punc_cat = {"Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"}
    # Comment out features that give us worse accuracy
    # if all(unicodedata.category(x) in punc_cat for x in word):
    #     features.append("PUNCTUATION")
    # if i < len(sent) - 1:
    #     features.append(sent[i + 1])
    # else:
    #     features.append(str('EOS'))

    return features


def sent2features(sent, index):
    """Call word2features on sentence."""
    return word2features(sent, index)


def update_outputs(outputs_a, outputs_b, sent_a, sent_b, keep):
    """Return lists of swapped outputs."""
    if keep:
        outputs_a.append(sent_a)
        outputs_b.append(sent_b)
    else:
        outputs_a.append(sent_b)
        outputs_b.append(sent_a)
    return outputs_a, outputs_b


def plot_accuracies_all_models(accuracies_set_a: list, accuracies_set_b):
    data = {'Model': ['Baseline most frequent', 'Baseline mlp', 'Paradigm A default',
                      'Paradigm A train_opt', 'Paradigm A func', 'Paradigm A tuned',
                      'Paradigm A sklearn', 'Paradigm B default'],
            'Accuracy_dev_set1': accuracies_set_a,
            'Accuracy_dev_set2': accuracies_set_b
            }

    # 'Accuracy_dev_set1': [0.233, 0.922, 0.979, 0.985, 0.980, 0.984, 0.985, 0.952],
    # 'Accuracy_dev_set2': [0.172, 0.839, 0.969, 0.977, 0.966, 0.972, 0.974, 0.515]

    # 'Accuracy_test_set1': [0.238, 0.929, 0.982, 0.988, 0.984, 0.985, 0.989, 0.958],
    # 'Accuracy_test_set2': [0.171, 0, 0, 0, 0, 0, 0, 0]

    df = pd.DataFrame(data)
    df.plot(x='Model', y=['Accuracy_dev_set1', 'Accuracy_dev_set2'], rot=20)
    plt.savefig('accuracy_dev.png')
    plt.show()


def reformat_train_evaluate_set(directory, train_set, dev_set, test_set, set: str):
    # 1: EXTRACT AND REFORMAT DATA

    tagged_sents_train, tagged_sents_val, tagged_sents_test, sentences_train, sentences_val, sentences_test = \
        extract_and_reformat_data(directory, train_set,
                                  dev_set,
                                  test_set)  # 4274/14287 training set, 572/1654 validating set, 586/1721  test
    tokens_train, tokens_val, tokens_test = get_tokens_from_sentences(
        sentences_train, sentences_val, sentences_test)  # 48655/469366, 6644/55482, 6580/55603
    gold_sent_labels_train, gold_sent_labels_val, gold_sent_labels_test = get_sentence_gold_labels_from_datasets(
        tagged_sents_train, tagged_sents_val, tagged_sents_test)
    gold_tokens_train, gold_tokens_val, gold_tokens_test = get_token_gold_labels(
        gold_sent_labels_train, gold_sent_labels_val, gold_sent_labels_test)

    X_train, y_train, X_val, y_val = get_X_and_y_features_from_datasets(tagged_sents_train, tagged_sents_val)

    # 2: TRAIN, TUNE, EVALUATE

    all_accuracies = []
    # Baselines: basic and advanced
    accuracy_most_frequent = train_baseline(tokens_train, gold_tokens_train, tokens_val, gold_tokens_val,
                                            'most_frequent')  # 0.23344370860927152 // 0.17295699506146137
    all_accuracies.append(accuracy_most_frequent)
    accuracy_mlp = train_baseline(tokens_train, gold_tokens_train, tokens_val, gold_tokens_val,
                                  'mlp')  # Accuracy on the val set with strategy mlp: 0.922787477423239 // 0.8399120435456544
    all_accuracies.append(accuracy_mlp)
    # # Train basic model A
    model_outputs, tagger, accuracy, predicted_token_labels = train_model(tagged_sents_train, sentences_val,
                                                                          tagged_sents_val,
                                                                          'crf')  # Accuracy on the validation set with crf.CRFTagger default: 0.9793798916315473 // 0.9697018852961321
    all_accuracies.append(accuracy)
    print(classification_report(gold_tokens_val, predicted_token_labels, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(gold_tokens_val, predicted_token_labels, xticks_rotation='vertical')
    plt.savefig("confusion_A.png")
    plt.show()

    # Hyperparameter tuning

    # Experiment with training_opt
    if set == 'spa':
        dataframe, best_accuracy, best_parameters = tune_training_opt(tagged_sents_train, tagged_sents_val,
                                                                      gold_sent_labels_val, 'spa')  # 0.9775783136873221
    else:
        dataframe, best_accuracy, best_parameters = tune_training_opt(tagged_sents_train, tagged_sents_val,
                                                                      gold_sent_labels_val, 'en')  # 0.9850993377483444
    all_accuracies.append(best_accuracy)
    # Specify function to get features
    model_outputs_w_f, tagger_with_func, accuracy_w_f, predicted_token_labels_w_f = train_model(tagged_sents_train,
                                                                                                sentences_val,
                                                                                                tagged_sents_val,
                                                                                                'crf_func')  # Accuracy on the validation set with crf_func, model A : 0.9804334738109572 // 0.966367470530983
    all_accuracies.append(accuracy_w_f)

    # Combine training options and feature function
    model_outputs_tuned, tagger_tuned, accuracy_tuned, predicted_token_labels_tuned = train_best_params_n_function(
        tagged_sents_train,
        best_parameters,
        sentences_val,
        tagged_sents_val,
        'crf_tuned')  # Accuracy on the validation set with crf_tuned, model A 0.984 :  // 0.9723153455174651
    all_accuracies.append(accuracy_tuned)

    # Train model on sklearn crf implementation
    accuracy_sklearn, pred_b, labels = train_crf_sklearn(X_train, X_val, y_train,
                                                         y_val)  # The highest accuracy (0.9850993377483444) was achieved with the algorithm 'ap' // 0.97433401824015
    all_accuracies.append(accuracy_sklearn)

    # Linguistic Error analysis
    perform_error_analysis(model_outputs,
                           tagged_sents_val)  # en = 137 incorrectly labeled tokens, 63 of which are unique. that, nonstop, which, to, is // 1681 incorrectly labeled tokens, 1135 of which are unique. que, como, la, cuando, mientras
    if set == 'en':
        get_token_previous_posterior_with_labels(tagged_sents_train, tagged_sents_val, 'which')
        get_token_previous_posterior_with_labels(tagged_sents_train, tagged_sents_val, 'that')
    elif set == 'spa':
        get_token_previous_posterior_with_labels(tagged_sents_train, tagged_sents_val, 'que')
        get_token_previous_posterior_with_labels(tagged_sents_train, tagged_sents_val, 'como')

    # Performance evaluation: statistical significance testing
    # Train model B
    model_outputs_b, tagger_b, accuracy_b, predicted_token_labels_b = train_model(tagged_sents_train, sentences_val,
                                                                                  tagged_sents_val,
                                                                                  'hmm')  # 0.9528898254063817 // 0.5158609999639523
    all_accuracies.append(accuracy_b)
    paired_randomization_test(predicted_token_labels, predicted_token_labels_b, gold_tokens_val, rejection_level=0.05,
                              R=1000)  # p-value is 0.000999000999000999 Reject = True // same

    return all_accuracies


def reformat_train_evaluate_set_final(directory, train_set, dev_set, test_set, set: str):
    # 1: EXTRACT AND REFORMAT DATA

    tagged_sents_train, tagged_sents_val, tagged_sents_test, sentences_train, sentences_val, sentences_test = \
        extract_and_reformat_data(directory, train_set,
                                  dev_set,
                                  test_set)  # 4274/14287 training set, 572/1654 validating set, 586/1721  test
    tokens_train, tokens_val, tokens_test = get_tokens_from_sentences(
        sentences_train, sentences_val, sentences_test)  # 48655/469366, 6644/55482, 6580/55603
    gold_sent_labels_train, gold_sent_labels_val, gold_sent_labels_test = get_sentence_gold_labels_from_datasets(
        tagged_sents_train, tagged_sents_val, tagged_sents_test)
    gold_tokens_train, gold_tokens_val, gold_tokens_test = get_token_gold_labels(
        gold_sent_labels_train, gold_sent_labels_val, gold_sent_labels_test)
    X_train, y_train, X_test, y_test = get_X_and_y_features_from_datasets(tagged_sents_train, tagged_sents_test)
    all_accuracies = []
    # Baselines: basic and advanced
    accuracy_most_frequent = train_baseline(tokens_train, gold_tokens_train, tokens_test, gold_tokens_test,
                                            'most_frequent')  # 0.23814589665653496 // 0.171447583763466
    all_accuracies.append(accuracy_most_frequent)
    accuracy_mlp = train_baseline(tokens_train, gold_tokens_train, tokens_test, gold_tokens_test,
                                  'mlp')  # 0.9291793313069909 // 0.837832491052641
    all_accuracies.append(accuracy_mlp)
    # # Train basic model A
    model_outputs, tagger, accuracy, predicted_token_labels = train_model(tagged_sents_train, sentences_test,
                                                                          tagged_sents_test,
                                                                          'crf')  # 0.9828267477203647 // 0.9684189701994497
    all_accuracies.append(accuracy)
    print(classification_report(gold_tokens_test, predicted_token_labels, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(gold_tokens_test, predicted_token_labels, xticks_rotation='vertical')
    plt.savefig("confusion_A_test.png")
    plt.show()

    # Hyperparameter tuning

    # Experiment with training_opt
    if set == 'spa':
        dataframe, best_accuracy, best_parameters = tune_training_opt(tagged_sents_train, tagged_sents_test,
                                                                      gold_sent_labels_test,
                                                                      'spa')  # 0.9741201014333759
    else:
        dataframe, best_accuracy, best_parameters = tune_training_opt(tagged_sents_train, tagged_sents_test,
                                                                      gold_sent_labels_test,
                                                                      'en')  # 0.988145896656535
    all_accuracies.append(best_accuracy)
    # Specify function to get features
    model_outputs_w_f, tagger_with_func, accuracy_w_f, predicted_token_labels_w_f = train_model(tagged_sents_train,
                                                                                                sentences_test,
                                                                                                tagged_sents_test,
                                                                                                'crf_func')  # 0.9848024316109423 // 0.9651457655162491
    all_accuracies.append(accuracy_w_f)

    # Combine training options and feature function
    model_outputs_tuned, tagger_tuned, accuracy_tuned, predicted_token_labels_tuned = train_best_params_n_function(
        tagged_sents_train,
        best_parameters,
        sentences_test,
        tagged_sents_test,
        'crf_tuned')  # 0.9857142857142858 :  // 0.9727352840674064
    all_accuracies.append(accuracy_tuned)

    # Train model on sklearn crf implementation
    accuracy_sklearn, pred_b, labels = train_crf_sklearn(X_train, X_test, y_train,
                                                         y_test)  # (0.9899696048632218) 'ap' // (0.9727352840674064)
    all_accuracies.append(accuracy_sklearn)

    # Linguistic Error analysis
    perform_error_analysis(model_outputs,
                           tagged_sents_test)
    if set == 'en':  # en = 113 incorrectly labeled tokens, 73 of which are unique: much, is, like, first, night
        get_token_previous_posterior_with_labels(tagged_sents_train, tagged_sents_test, 'much')  # 8 times
        get_token_previous_posterior_with_labels(tagged_sents_train, tagged_sents_test, 'is')  # 7 times
    elif set == 'spa':  # 1756 incorrectly labeled tokens, 1173 of which are unique: que, la, cuando, como, todo_
        get_token_previous_posterior_with_labels(tagged_sents_train, tagged_sents_test, 'que')  # 155 times
        get_token_previous_posterior_with_labels(tagged_sents_train, tagged_sents_test, 'la')  # 23 times

    # Performance evaluation: statistical significance testing
    # Train model B
    model_outputs_b, tagger_b, accuracy_b, predicted_token_labels_b = train_model(tagged_sents_train, sentences_test,
                                                                                  tagged_sents_test,
                                                                                  'hmm')  # 0.9586626139817629
    all_accuracies.append(accuracy_b)
    paired_randomization_test(predicted_token_labels, predicted_token_labels_b, gold_tokens_test,
                              rejection_level=0.05,
                              R=1000)  # p-value is 0.000999000999000999 Reject = True // same
    return all_accuracies


if __name__ == '__main__':
    # VALIDATING SET
    # Dataset 1: Atis (English)
    accuracies_atis = reformat_train_evaluate_set(
        'UD_English-Atis-master/',
        'en_atis-ud-train.conllu',
        'en_atis-ud-dev.conllu',
        'en_atis-ud-test.conllu',
        'en'
    )

    # Dataset 2: AnCora (Spanish), that is much larger
    accuracies_ancora = reformat_train_evaluate_set(
        'UD_Spanish-AnCora-master/',
        'es_ancora-ud-train.conllu',
        'es_ancora-ud-dev.conllu',
        'es_ancora-ud-test.conllu',
        'spa'
    )

    plot_accuracies_all_models(accuracies_atis, accuracies_ancora)

    # TEST SET

    # Dataset 1: Atis (English)
    accuracies_atis = reformat_train_evaluate_set_final(
        'UD_English-Atis-master/',
        'en_atis-ud-train.conllu',
        'en_atis-ud-dev.conllu',
        'en_atis-ud-test.conllu',
        'en',
    )

    # Dataset 2: AnCora (Spanish), that is much larger
    accuracies_ancora = reformat_train_evaluate_set_final(
        'UD_Spanish-AnCora-master/',
        'es_ancora-ud-train.conllu',
        'es_ancora-ud-dev.conllu',
        'es_ancora-ud-test.conllu',
        'spa',
    )

    plot_accuracies_all_models(accuracies_atis, accuracies_ancora)
