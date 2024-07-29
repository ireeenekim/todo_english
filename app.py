from flask import Flask, request, render_template, redirect, url_for, send_file
from pydub import AudioSegment
import speech_recognition as sr
from transformers import pipeline, set_seed, GPT2LMHeadModel, GPT2Tokenizer
from diffusers import StableDiffusionPipeline
from keybert import KeyBERT
import torch
import os
import subprocess

app = Flask(__name__)

# Ensure ffmpeg is installed
def install_ffmpeg():
    try:
        subprocess.check_call(['ffmpeg', '-version'])
    except subprocess.CalledProcessError:
        print("ffmpeg is not installed. Please install it using 'brew install ffmpeg' or your system's package manager.")

install_ffmpeg()

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Initialize the KeyBERT model
kw_model = KeyBERT()

# Initialize the image generation model with cartoon-like style
pipe = StableDiffusionPipeline.from_pretrained(
    "Nitrosocke/Arcane-Diffusion", 
    torch_dtype=torch.float32,
    use_auth_token=True,  # Add this line if you need to use a Hugging Face authentication token
    force_download=True  # This line ensures that the model files are re-downloaded
)
device = "cpu"
pipe = pipe.to(device)

# Initialize the GPT-2 model for generating comments
comments_model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def generate_image_from_text(prompt):
    # Generate image from text prompt
    image = pipe(prompt, negative_prompt="text, letters, words").images[0]
    # Ensure the static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')
    image_path = "static/generated_image.png"
    image.save(image_path)
    return image_path

def voice_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)

def process_text(conversation_text):
    summary = summarizer(conversation_text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
    return summary

def generate_hashtags(summary):
    keywords = kw_model.extract_keywords(summary, keyphrase_ngram_range=(1, 2), stop_words=None)
    hashtags = [f"#{keyword[0].replace(' ', '')}" for keyword in keywords[:4]]
    return hashtags

def generate_comments(summary):
    set_seed(42)
    prompt = f"Comments about: {summary}\n1."
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    comments = []
    for _ in range(4):
        comment = comments_model.generate(input_ids, max_length=50, num_return_sequences=1, truncation=True)
        comment_text = tokenizer.decode(comment[0], skip_special_tokens=True)
        comments.append(comment_text.split('\n')[1].strip())

    return comments

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        audio_file_path = "uploaded_audio.wav"
        file.save(audio_file_path)
        conversation_text = voice_to_text(audio_file_path)
        processed_text = process_text(conversation_text)
        image_path = generate_image_from_text(processed_text)
        hashtags = generate_hashtags(processed_text)
        comments = generate_comments(processed_text)
        
        return render_template('result.html', summary=processed_text, image_path=image_path, hashtags=hashtags, comments=comments)

@app.route('/image/<filename>')
def image(filename):
    return send_file(os.path.join("static", filename))

if __name__ == "__main__":
    app.run(debug=True)
