from flask import Flask, render_template, request
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# create a flask app
app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

#Clean up the sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#Converts the sentences into a bag of words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence) #bow: Bag Of Words, feed the data into the neural network
    res = model.predict(np.array([bow]))[0] #res: result. [0] as index 0
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result



@app.route("/")
def home():
    return render_template("index.html")


# define a route for the chat page
@app.route("/chat")
def chat():

    essage = input("You: ")
    

    # get the user input from the query string
    user_input = request.args.get("user_input")
    # get the bot response

    ints = predict_class(user_input)
    bot_response = get_response(ints, intents)
    return {"bot_response": str(bot_response)}

# run the app
if __name__ == "__main__":
    app.run(debug=True)