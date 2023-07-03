import random
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop

#Get skakespeare poetic quotes
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

#read file, decode, and convert to lowercase so that we get a better performance later
text = open(filepath, "rb").read().decode(encoding='utf-8').lower()

#select a part of the text, the whole dataset might take a while for the machine
text = text[300000:800000]

#filter-out unique characters, and sort it
characters = sorted(set(text))

#create two dictionaries to convert into numerical format, and vice versa

#dictionary with the character as the key, and index as value, for all incides and characters in the enumaration of the characters
#enumaration - assigns one number to each character in the set
char_to_index = dict((c,i) for i, c in enumerate(characters))

index_to_char = dict((i,c) for i, c in enumerate(characters))


SEQ_LENGTH = 40

#"how many characters to shift to next sequence"?
STEP_SIZE = 3

'''

#sentence - a sentence, next_characters - the last character needed to finish a sentence
sentences = []
next_characters = []


#add to sentences
for i in range(0, len(text)- SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i : i+SEQ_LENGTH])
    next_characters.append(text[i+SEQ_LENGTH])
    
#numoy array full of zeros, boolean
#length of sentences BY the sequence length BY the length of characters
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

#Training data
#assign index to all sentences
#for each sentences, enumarate every character, and index it
#sentence number i at position t at character any x exists, set to 1(true)

for i,sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

#feed the arrays into a recurrent neural network, then use it to predict the approptiate quote
model = Sequential()
#LSTM - long short term memory - memory of the network, remembers the important/relevant characters
model.add(LSTM(128, input_shape = (SEQ_LENGTH, len(characters))))

#Dense - add complexity to the layer. Layer with the number of characters
model. add(Dense(len(characters)))

#Softmax activation - scales output so that all values add up to 1. 
model.add(Activation('softmax'))

#compile model
model.compile(loss = 'categorical_crossentropy', optimizer=RMSprop(lr=0.01))

#fit into the training data
#batch_size - how many examples to put at into network once
#epochs - how many times out network is going to see the same data again
model.fit(x, y, batch_size=256, epochs=4)

#save, and load it later instead of training it again and again
model.save('textgenerator.model')

'''

#run the commented-out code above first to generate the model, and then import it

model = tf.keras.models.load_model('textgenerator.model')


#prediction sample depending on the temperature
#THe higher the temp, the more the creativity of the poetic text
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1,preds, 1)
    return np.argmax(probas)


def generate_text(length, temperature):
    start_index = random.randint(0,len(text)-SEQ_LENGTH)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    
    for i in range(length):
        x = np.zeros((1,SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1
    
        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]
        
        generated += next_character
        
        sentence = sentence[1:] + next_character
        
    return generated


print('---------0.2---------')
print(generate_text(300,0.2))


print('---------0.4---------')
print(generate_text(300,0.4))


print('---------0.6---------')
print(generate_text(300,0.6))



print('---------0.8---------')
print(generate_text(300,0.8))


print('---------1.0---------')
print(generate_text(300,1.0))
