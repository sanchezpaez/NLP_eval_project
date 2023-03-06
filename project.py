# Sandra Sanchez
# NLP Eval project

from pathlib import Path

from conllu import parse
from nltk.tag import hmm, crf, perceptron
from sklearn import metrics
from sklearn.dummy import DummyClassifier


def read_datasets(corpus_dirname, train_filename, dev_filename,  test_filename):
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
    tagged_train = [[(word['form'], word['upos']) for word in sent] for sent in train_set]
    tagged_val = [[(word['form'], word['upos']) for word in sent] for sent in val_set]
    tagged_test = [[(word['form'], word['upos']) for word in sent] for sent in test_set]

    return tagged_train, tagged_val, tagged_test


def get_tokens_from_datasets(train_set, val_set, test_set):
    tokens_train = [[(word['form'], '') for word in sent] for sent in train_set]
    tokens_val = [[word['form'] for word in sent] for sent in val_set]
    tokens_test = [[word['form'] for word in sent] for sent in test_set]

    return tokens_train, tokens_val, tokens_test


def get_gold_labels_from_datasets(tagged_train, tagged_val, tagged_test):
    gold_labels_train = [[(word[1]) for word in sent] for sent in tagged_train]
    gold_labels_val = [[(word[1]) for word in sent] for sent in tagged_val]
    gold_labels_test = [[(word[1]) for word in sent] for sent in tagged_test]

    return gold_labels_train, gold_labels_val, gold_labels_test


def train_model(tagged_trainset, tokens_valset, tagged_valset):
    # Train model

    # Either this one
    # trainer = hmm.HiddenMarkovModelTrainer()
    # tagger = trainer.train_supervised(tagged_train)

    # Or this one
    tagger = crf.CRFTagger()
    tagger.train(tagged_trainset, 'model.crf.tagger')

    # print(tagger.tag("i would like a flight from atlanta to boston .".split()))

    model_outputs = tagger.tag_sents(tokens_valset)
    print(f"Predicted tagged val sentence {model_outputs[0]}")

    accuracy = tagger.evaluate(tagged_valset)
    print("Accuracy on the validation set:", accuracy)  # 0.9528898254063817
    predicted_labels = [tag for sentence in model_outputs for token, tag in sentence]
    # predicted_labels = [[tag for token, tag in sentence] for sentence in model_outputs]

    return model_outputs, tagger, accuracy, predicted_labels


def get_gold_labels(gold_labels_train, gold_labels_val, gold_labels_test):
    gold_train = [label for sentence  in gold_labels_train for label in sentence]
    gold_val = [label for sentence  in gold_labels_val for label in sentence]
    gold_test = [label for sentence  in gold_labels_test for label in sentence]
    return gold_train, gold_val, gold_test


def train_baseline(tokens_train, gold_train, tokens_val, gold_val):
    classifier = DummyClassifier(strategy='most_frequent')
    tokens_train = [token for sentence  in tokens_train for token in sentence]
    tokens_val = [token for sentence  in tokens_val for token in sentence]
    # print(len(tokens_train), len(gold_train))  # 4274 48655
    # tokens_train = [t for sentence in tokens_train for t in sentence]  # [6644, 572]
    classifier.fit(tokens_train, gold_train)
    binary_balanc_predictions = classifier.predict(tokens_val)
    binary_balanc_probs = classifier.predict_proba(tokens_val)
    print(f'Accuracy on the val set with dummy classifier: {classifier.score(tokens_val, gold_val)}')


train_data, val_data, test_data = read_datasets('UD_English-Atis-master/', 'en_atis-ud-train.conllu', 'en_atis-ud-dev.conllu', 'en_atis-ud-test.conllu')
# print(train_data[0])
# print(train_data[0][0])
tagged_train, tagged_val, tagged_test = tag_datasets(train_data, val_data, test_data)
# print(f"Tagged val sent: {tagged_val[:3]}")
tokens_train, tokens_val, tokens_test = get_tokens_from_datasets(train_data, val_data, test_data)
gold_labels_train, gold_labels_val, gold_labels_test = get_gold_labels_from_datasets(tagged_train, tagged_val, tagged_test)
model_outputs, tagger, accuracy, predicted_labels = train_model(tagged_train, tokens_val, tagged_val)
gold_train, gold_val, gold_test = get_gold_labels(gold_labels_train, gold_labels_val, gold_labels_test)
print(metrics.classification_report(gold_val, predicted_labels, zero_division=0))
# print(len(gold_val), len(predicted_labels))  # 572 6644
train_baseline(tokens_train, gold_train, tokens_val, gold_val)

##########################

# Now try with a dataset in Spanish, that is much larger

# train_data, val_data, test_data = read_datasets('UD_Spanish-AnCora-master/', 'es_ancora-ud-train.conllu', 'es_ancora-ud-dev.conllu', 'es_ancora-ud-test.conllu')
# tagged_train, tagged_val, tagged_test = tag_datasets(train_data, val_data, test_data)
# tokens_train, tokens_val, tokens_test = get_tokens_from_datasets(train_data, val_data, test_data)
# gold_labels_train, gold_labels_val, gold_labels_test = get_gold_labels_from_tagged_data(tagged_train, tagged_val, tagged_test)
# model_outputs, tagger, accuracy, predicted_labels = train_model(tagged_train, tokens_val, tagged_val)
# print(metrics.classification_report(gold_labels_val, predicted_labels, zero_division=0))

