import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from googletrans import Translator
import pyttsx3


# Example dataset for different festivals with sections
holiday_songs = {
    'Diwali': [
        "main: Diya jale, jingle bells, lights all around",
        "chorus: Happy Diwali, colors bright, joy unbound",
        "ending: Lights shine bright, Diwali in the night",
    ],
    'Pongal': [
        "main: Harvest time, sweet treats, Pongal is here",
        "chorus: Celebrate with love, family near and dear",
        "ending: Pongal joy, happiness all year",
    ],
    'Thanksgiving': [
        "main: Turkey on the table, gratitude in the air",
        "chorus: Family gathers around, Thanksgiving cheer",
        "ending: Thankful hearts, memories so dear",
    ],
    'Easter': [
        "main: Easter eggs, bunnies hop, spring is in bloom",
        "chorus: Joyful songs, new beginnings, chasing away the gloom",
        "ending: Easter joy, flowers in the room",
    ],
    'Ramzan': [
        "main: Ramzan nights, moonlight, hearts full of faith",
        "chorus: Eid Mubarak, peace and love in every breath",
        "ending: Faith and love, Ramzan fest",
    ],
    'Christmas': [
        "main: Jingle bells, jingle bells, jingle all the way",
        "chorus: Oh what fun it is to ride in a one-horse open sleigh",
        "ending: Merry Christmas, joy today",
    ],
    'Halloween': [
        "main: Ghosts and goblins, spooks galore, scary witches at your door",
        "chorus: Jack-o-lanterns shining bright, wishing you a haunting night",
        "ending: Spooky fun, Halloween delight",
    ],
    'New Year': [
        "main: New year, new beginnings, fresh start, new innings",
        "chorus: Celebrate with cheer, wishing you a happy new year",
        "ending: Cheers to health, happiness here",
    ],
    'Hanukkah': [
        "main: Eight nights, candles bright, Hanukkah is here",
        "chorus: Spin the dreidel, gifts and cheer",
        "ending: Hanukkah joy, all clear",
    ],
    'Valentine\'s Day': [
        "main: Roses red, love is in the air",
        "chorus: Valentine's Day magic, feelings we share",
        "ending: Love and hearts, everywhere",
    ],
    'Holi': [
        "main: Colors fly, joy and laughter fill the sky",
        "chorus: Happy Holi to all, let's dance and celebrate high",
        "ending: Colors bright, happiness nigh",
    ],
    'Navratri': [
        "main: Nine nights of dance, Garba in the air",
        "chorus: Navratri is here, with blessings and care",
        "ending: Dancing joy, Navratri fair",
    ],
    'Ganesh Chaturthi': [
        "main: Ganapati Bappa Morya, let's celebrate",
        "chorus: Ganesh Chaturthi joy, no more wait",
        "ending: Blessings flow, Ganesh's grace",
    ],
    'Onam': [
        "main: Pookalam bright, Onam delight",
        "chorus: King Mahabali returns, happiness in sight",
        "ending: Onam joy, pure and bright",
    ],
   
    # Add other festivals here...
}

# Prepare data for the model
def prepare_data(songs):
    combined_songs = []
    for festival, lyrics in songs.items():
        combined_songs.extend(lyrics)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(combined_songs)
    vocab_size = len(tokenizer.word_index) + 1

    sequences = []
    for song in combined_songs:
        token_sequence = tokenizer.texts_to_sequences([song])[0]
        for i in range(1, len(token_sequence)):
            n_gram_sequence = token_sequence[:i + 1]
            sequences.append(n_gram_sequence)

    max_seq_len = max([len(x) for x in sequences])
    sequences = np.array(pad_sequences(sequences, maxlen=max_seq_len, padding='pre'))
    predictors, label = sequences[:, :-1], sequences[:, -1]
    label = tf.keras.utils.to_categorical(label, num_classes=vocab_size)

    return predictors, label, max_seq_len, vocab_size, tokenizer


predictors, label, max_seq_len, vocab_size, tokenizer = prepare_data(holiday_songs)

model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_seq_len - 1))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(150))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(predictors, label, epochs=100, verbose=1)

# Function to generate lyrics
def generate_lyrics(festival, seed_text, next_words, max_seq_len, model, tokenizer):
    if festival not in holiday_songs:
        return "Festival not recognized. Please enter a valid festival."

    generated_text = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        output_word = tokenizer.index_word[np.argmax(predicted)]
        generated_text += " " + output_word

    return generated_text

# Function to translate text
def translate_text(text, target_language):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

# Function for Text-to-Speech
def text_to_speech(text, language="en"):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("voice", language)
    engine.say(text)
    engine.runAndWait()
