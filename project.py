# Sandra Sanchez
# NLP Eval project

from pathlib import Path

from conllu import parse
from nltk.tag import hmm, crf, perceptron
from sklearn import metrics
from sklearn.dummy import DummyClassifier

CORPUS_PATH = Path('UD_English-Atis-master/')

TRAIN_PATH = CORPUS_PATH / 'en_atis-ud-train.conllu'
VAL_PATH = CORPUS_PATH / 'en_atis-ud-dev.conllu'
TEST_PATH = CORPUS_PATH / 'en_atis-ud-test.conllu'

with open(TRAIN_PATH, 'r') as file:
    train_data = parse(file.read())

with open(VAL_PATH, 'r') as file:
    val_data = parse(file.read())

with open(TEST_PATH, 'r') as file:
    test_data = parse(file.read())

# print(train_data[0])
# print(train_data[0][0])

tagged_train = [[(word['form'], word['upos']) for word in sent] for sent in train_data]
tagged_val = [[(word['form'], word['upos']) for word in sent] for sent in val_data]
tagged_test = [[(word['form'], word['upos']) for word in sent] for sent in test_data]

print(f"Tagged val sent: {tagged_val[0]}")

tokens_train = [[(word['form'], '') for word in sent] for sent in train_data]
tokens_val = [[word['form'] for word in sent] for sent in val_data]
tokens_test = [[word['form'] for word in sent] for sent in test_data]

# Gold labels
gold_labels_val = [tag for sentence in tagged_val for token,tag in sentence]
gold_labels_train = [tag for sentence in tagged_train for token,tag in sentence]

# Train model
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(tagged_train)

tagger = crf.CRFTagger()
tagger.train(tagged_train, 'model.crf.tagger')

# Train baseline
# classifier = DummyClassifier(strategy='most_frequent')
# classifier.fit(tokens_train, gold_labels_train)
# binary_balanc_predictions = classifier.predict(tokens_val)
# binary_balanc_probs = classifier.predict_proba(tokens_val)
# print(f'Accuracy on the val set with dummy classifier: {classifier.score(tokens_val, gold_labels_val)}')

# print(tagger.tag("i would like a flight from atlanta to boston .".split()))

model_outputs = tagger.tag_sents(tokens_val)
print(f"Predicted tagged val sentence {model_outputs[0]}")

accuracy = tagger.evaluate(tagged_val)
print("Accuracy on the validation set:", accuracy)  # 0.9528898254063817

# gold_labels = []
# for sentence in tagged_val:
#     for word in sentence:
#         gold_labels.append(word[1])

# predicted_labels = []
# for sentence in model_outputs:
#     for word in sentence:
#         predicted_labels.append(word[1])


predicted_labels = [tag for sentence in model_outputs for token, tag in sentence]

print(metrics.classification_report(gold_labels_val, predicted_labels, zero_division=0))


##########################

# Now try with a dataset in Spanish, that is much larger

CORPUS_PATH = Path('UD_Spanish-AnCora-master-Atis-master/')

TRAIN_PATH = CORPUS_PATH / 'es_ancora-ud-train.conllu'
VAL_PATH = CORPUS_PATH / 'es_ancora-ud-dev.conllu'
TEST_PATH = CORPUS_PATH / 'es_ancora-ud-test.conllu'

with open(TRAIN_PATH, 'r') as file:
    train_data = parse(file.read())

with open(VAL_PATH, 'r') as file:
    val_data = parse(file.read())

with open(TEST_PATH, 'r') as file:
    test_data = parse(file.read())
