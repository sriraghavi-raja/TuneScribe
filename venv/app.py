from flask import Flask, request, render_template, send_file
from data_model import generate_lyrics, max_seq_len, model, tokenizer, translate_text, text_to_speech
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
@app.route('/generate', methods=['POST'])
def generate():
    festival = request.form['festival']
    seed_text = request.form['seed_text']
    next_words = int(request.form['next_words'])
    melody_style = request.form.get('melody_style', 'default')
    language = request.form.get('language', 'en')

    
    generated_text = generate_lyrics(festival, seed_text, next_words, max_seq_len, model, tokenizer)
    translated_text = translate_text(generated_text, language)
    
   
    if 'play_voice' in request.form:
        text_to_speech(translated_text, language)

    
    lyrics_file_path = None
    if 'download' in request.form:
        filename = "generated_lyrics.txt"
        with open(filename, "w", encoding="utf-8") as file:
            file.write(translated_text)
        lyrics_file_path = filename 

    return render_template('index.html', generated_text=translated_text, lyrics_file_path=lyrics_file_path)

if __name__ == '__main__':
    app.run(debug=True, port=8081)
