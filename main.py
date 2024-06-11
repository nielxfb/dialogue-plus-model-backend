from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

import pickle
import string

app = Flask(__name__)
CORS(app, supports_credentials=True)

classifier = None

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'

def preprocess_words(words):
    words = [word for word in words if word.lower() not in stopwords.words('english')]
    words = [word for word in words if word.lower() not in string.punctuation]
    words = [word for word in words if word.isalpha()]

    word_tag = pos_tag(words)

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in word_tag]

    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return words

@app.route("/classify", methods=["POST"])
def get_category():
    text = request.json['text']

    words = word_tokenize(text)
    words = preprocess_words(words)
    feature = FreqDist(words)
    category = classifier.classify(feature)
    if category == 0:
        category = 'Not hateful'
    else:
        category = 'Hateful'

    return jsonify({"category": category})

if __name__ == "__main__":
    classifier = pickle.load(open('model.pickle', 'rb'))
    app.run(port=5000, debug=True)