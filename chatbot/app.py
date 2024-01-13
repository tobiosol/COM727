from flask import Flask, render_template, request, url_for, flash, redirect, abort,send_file
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import sqlite3
import os
import training

# create a flask app
app = Flask(__name__)

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('model/intents.json').read())

words = pickle.load(open('model/words.pkl', 'rb'))
classes = pickle.load(open('model/classes.pkl', 'rb'))
model = load_model('model/chatbot_model.keras')

def load_resource():
    words = pickle.load(open('model/words.pkl', 'rb'))
    classes = pickle.load(open('model/classes.pkl', 'rb'))
    model = load_model('model/chatbot_model.keras')

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



def get_db_connection():
    conn = sqlite3.connect('db/database.db')
    conn.row_factory = sqlite3.Row
    return conn


def generate_intent_content(tag, pattern, response):
    pattern_list = []
    pattern_list = pattern.split("~")
    response_list = response.split("~")
    intent_dict = {
        "tag": tag,
        "patterns": pattern_list,
        "responses": response_list
    }

    print(f'intent_dict {intent_dict}')
    return intent_dict

@app.route("/admin", methods=["GET", "POST"])
def admin():
    print(f'{request}')
    if request.method == 'POST':
        tag = request.form['tag']
        pattern = request.form['question']
        response = request.form['answer']

        if not tag.strip:
            flash('Tag is required!')
        elif not pattern.strip:
            flash('Pattern is required!')
        elif not response.strip:
            flash('Response is required!')
        else:
            conn = get_db_connection()
            conn.execute('INSERT INTO intent (tag, pattern, response) VALUES (?, ?, ?)',  (tag, pattern, response))
            conn.commit()
            conn.close()
            generate_intent_content(tag, pattern, response)

    conn = get_db_connection()
    intents = conn.execute('SELECT * FROM intent ORDER BY created DESC LIMIT 10').fetchall()
    conn.close()
    return render_template('admin.html', intents=intents)



@app.route("/generatefile", methods=["GET", "POST"])
def generate_file():
    if request.method == "POST":
        # get the file name and content from the form
        file_name = 'model/intents.json'
        intents  = []
        # read from database
        conn = get_db_connection()
        data = conn.execute('SELECT * FROM intent').fetchall()
        conn.close()
        # loop over each rows
        # iterate over the data using a for loop
        for id,created,tag,pattern,response in data:
            intent = generate_intent_content(tag, pattern, response)
            intents.append(intent)

        # convert to json
        json_content = {"intents": intents}
        # write to file
    
        with open(file_name, "w") as f:
            # write the json string to the file
            f.write(json.dumps(json_content))
        return {"file_name": file_name}
    
    return render_template("generatefile.html")

@app.route("/train_model", methods=["GET", "POST"])
def train_model():
    if request.method == "POST":
        training.train_data_model()
        load_resource()
    
    return redirect(url_for('admin'))
    # return render_template("generatefile.html")


@app.route("/download/<file_name>")
def download(file_name):
    # check if the file exists
    if os.path.exists(file_name):
        # send the file as an attachment
        return send_file(file_name, as_attachment=True)
    # return a 404 error if the file is not found
    return "File not found.", 404



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/home")
def l_home():
    return render_template("index.html")

# define a route for the chat page
@app.route("/chat")
def chat():
    # get the user input from the query string
    user_input = request.args.get("user_input")
    # get the bot response
    ints = predict_class(user_input)
    bot_response = get_response(ints, intents)
    return {"bot_response": str(bot_response)}



# run the app
if __name__ == "__main__":
    app.run(debug=True)