import random
import json
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout,Embedding, SimpleRNN
from tensorflow.keras.optimizers.legacy import SGD
from sklearn.model_selection import train_test_split
# from matplotlib import pyplot
# from keras.models import Sequential
# from keras.layers import LSTM, TimeDistributed, Dense,Reshape

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def train_data_model():
    print('train_data_model')
    lemmatizer = WordNetLemmatizer()
    intents = json.loads(open('model/intents.json').read())

    words = []
    classes = []
    documents = []
    ignore_letters=['?','!','.','/','@']

    def clean_non_english(txt):
        txt = re.sub(r'\W+', ' ', txt)
        txt = txt.lower()
        txt = txt.replace("[^a-zA-Z]", " ")
        return txt

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list,intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    stop_words = set(stopwords.words('english'))
    words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
    words = sorted(set(words))

    classes = sorted(set(classes))

    pickle.dump(words, open('model/words.pkl', 'wb'))
    pickle.dump(classes, open('model/classes.pkl', 'wb'))

    training = []

    output_empty = [0] * len(classes)

    for document in documents:
        bag=[]
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
   
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    # fit model
    
    history =  model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)
    model.save('model/chatbot_model.keras', history)

    # evaluate the model
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

train_data_model()

     # history =  model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)
    # model.save('model/chatbot_model.keras', history)

    # history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300, verbose=0)


    # model = Sequential()
    # model.add(Dense(128, input_shape=(len(train_x[0]),),activation='selu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='selu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='selu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='selu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(len(train_y[0]), activation='softmax'))


    

    
    # X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

    # model = Sequential()
    # model.add(Dense(128, input_shape=(len(train_x[0]),),activation='selu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='selu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='selu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='selu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(len(train_y[0]), activation='softmax'))

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    # Define the vocabulary size, embedding size, and hidden size
    # vocab_size = 6
    # embed_size = 128
    # hidden_size = 256

    # # Define the RNN model
    # model = Sequential()
    # # Add an embedding layer to convert words to vectors
    # model.add(Embedding(vocab_size, embed_size))
    # # Add a simple RNN layer with hidden_size units
    # model.add(SimpleRNN(hidden_size))
    # # Add a dense layer with vocab_size units and softmax activation
    # model.add(Dense(vocab_size, activation='softmax'))

    # # Compile the model with categorical crossentropy loss and Adam optimizer
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    # hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

    # kf = KFold(n_splits=k, shuffle=True, random_state=42)
    # # Loop over each fold
    # for train_index, test_index in kf.split(train_x):
    #     # Split the data into train and test sets
    #     X_train, X_test = train_x[train_index], train_x[test_index]
    #     y_train, y_test = train_y[train_index], train_y[test_index]

    #     # Fit the model on the train set
    #     model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)
    
    #     # Evaluate the model on the test set
    #     model.evaluate(np.array(X_test), np.array(y_test))


    # # Define input shape
    # input_shape = (93,)
    # # Create LSTM layer
    # lstm_layer = LSTM(32, activation='tanh', dropout=0.2)
    # # Wrap LSTM layer in TimeDistributed layer
    # time_distributed_layer = TimeDistributed(lstm_layer)
    # # Create dense layer for output
    # output_layer = Dense(1, activation='sigmoid')
    # # Connect layers in a sequential model
    # model = Sequential([Reshape((1, -1), input_shape=input_shape), time_distributed_layer, output_layer])
    # # Compile model
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    # X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

    # # Train model
    # history =  model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=32)
    # # Save model
    # model.save('model/lstm_model.h5', history)




    # plot loss during training
    # pyplot.subplot(211)
    # pyplot.title('Loss')
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    # pyplot.legend()
    # # plot accuracy during training
    # pyplot.subplot(212)
    # pyplot.title('Accuracy')
    # pyplot.plot(history.history['accuracy'], label='train')
    # pyplot.plot(history.history['val_accuracy'], label='test')
    # pyplot.legend()
    # pyplot.show()

    # #print('Done')

