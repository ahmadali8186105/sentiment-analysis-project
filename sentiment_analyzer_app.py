import customtkinter as ctk
import threading
import subprocess
import sys
import pickle
import re
import sounddevice as sd
import numpy as np
import whisper
import pyttsx3
from scipy.io.wavfile import write
from scipy.sparse import hstack
import time
import gc
import os

# ===========================================================
#  ML MODEL LOADING
# ===========================================================
print("Loading ML model and vectorizer...")

try:
    # Try to load enhanced model first
    try:
        clf = pickle.load(open("models/sentiment_lr_model_enhanced.pkl", "rb"))
        scaler = pickle.load(open("models/feature_scaler.pkl", "rb"))
        feature_info = pickle.load(open("models/feature_info.pkl", "rb"))
        enhanced_model = True
        print("Enhanced model with statistical features loaded.")
    except FileNotFoundError:
        # Fallback to basic model
        clf = pickle.load(open("models/sentiment_lr_model.pkl", "rb"))
        enhanced_model = False
        print("Basic model loaded.")
    
    vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
    label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
    print("ML components loaded successfully.")
except Exception as e:
    print(f"Error loading ML components: {e}")
    sys.exit(1)

# ================================
# Whisper model loading
# ================================
print("Loading Whisper (speech-to-text) model...")
try:
    whisper_model = whisper.load_model("small")
    print("Whisper model loaded successfully.\n")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    sys.exit(1)

# ===========================================================
#  AUDIO RECORDING
# ===========================================================
def record_audio(duration=5, fs=16000):
    print(f"\n🎙️ Recording for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write("temp/input_audio.wav", fs, audio)
    print("Recording saved as input_audio.wav\n")
    return "temp/input_audio.wav"

# ===========================================================
#  TEXT CLEANING AND FEATURE EXTRACTION
# ===========================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove mentions
    text = re.sub(r"@\w+", "", text)
    # Remove hashtags
    text = re.sub(r"#\w+", "", text)
    # Remove non-alphanumeric except punctuation
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    # Collapse repeated spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_statistical_features(text):
    """
    Extract statistical features from a single text
    """
    if not text or not isinstance(text, str):
        text = "emptytext"
    
    features = {}
    
    # Basic text statistics
    features['text_length'] = len(text)
    words = text.split()
    features['word_count'] = len(words)
    features['char_count'] = len(text)
    features['sentence_count'] = len(re.split(r'[.!?]+', text)) if text.strip() else 1
    
    # Average word length
    features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
    
    # Punctuation features
    features['punctuation_count'] = len(re.findall(r'[^\w\s]', text))
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['comma_count'] = text.count(',')
    features['period_count'] = text.count('.')
    
    # Uppercase features
    features['uppercase_count'] = sum(1 for c in text if c.isupper())
    features['uppercase_ratio'] = features['uppercase_count'] / (features['char_count'] + 1)
    
    # Numerical features
    features['digit_count'] = len(re.findall(r'\d', text))
    features['digit_ratio'] = features['digit_count'] / (features['char_count'] + 1)
    
    # Whitespace features
    features['whitespace_count'] = len(re.findall(r'\s', text))
    features['whitespace_ratio'] = features['whitespace_count'] / (features['char_count'] + 1)
    
    # Word-level statistics
    word_lengths = [len(word) for word in words] if words else [0]
    features['min_word_length'] = min(word_lengths)
    features['max_word_length'] = max(word_lengths)
    features['std_word_length'] = np.std(word_lengths) if len(word_lengths) > 1 else 0
    
    # Sentence-level statistics
    sentences = [sent.strip() for sent in re.split(r'[.!?]+', text) if sent.strip()]
    sentence_lengths = [len(sent.split()) for sent in sentences] if sentences else [0]
    features['min_sentence_length'] = min(sentence_lengths) if sentence_lengths else 0
    features['max_sentence_length'] = max(sentence_lengths) if sentence_lengths else 0
    features['mean_sentence_length'] = np.mean(sentence_lengths) if sentence_lengths else 0
    features['std_sentence_length'] = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
    
    # Vocabulary diversity features
    unique_words = set(words) if words else set()
    features['unique_word_count'] = len(unique_words)
    features['vocabulary_richness'] = len(unique_words) / (len(words) + 1) if words else 0
    
    # Readability features
    features['avg_words_per_sentence'] = features['word_count'] / (features['sentence_count'] + 1)
    features['complexity_score'] = features['avg_word_length'] * features['avg_words_per_sentence']
    
    # Handle inf/nan values
    for key, value in features.items():
        if np.isnan(value) or np.isinf(value):
            features[key] = 0
    
    return features

# ===========================================================
#  SENTIMENT PREDICTION
# ===========================================================
def predict_sentiment(text):
    cleaned = clean_text(text)
    
    # Get TF-IDF features
    tfidf_features = vectorizer.transform([cleaned])
    
    if enhanced_model:
        # Extract statistical features
        stat_features = extract_statistical_features(cleaned)
        
        # Convert to array in correct order
        stat_columns = feature_info['stat_columns']
        stat_array = np.array([[stat_features[col] for col in stat_columns]])
        
        # Scale statistical features
        stat_scaled = scaler.transform(stat_array)
        
        # Combine features
        combined_features = hstack([tfidf_features, stat_scaled])
        
        prediction = clf.predict(combined_features)[0]
    else:
        # Use only TF-IDF features for basic model
        prediction = clf.predict(tfidf_features)[0]
    
    sentiment_label = label_encoder.inverse_transform([prediction])[0]
    return sentiment_label

# ===========================================================
#  PIPELINE FUNCTION
# ===========================================================
def run_pipeline():
    # Step A: record audio
    audio_file = record_audio()

    # Step B: speech → text
    print("Converting speech to text...")
    result = whisper_model.transcribe(audio_file)
    spoken_text = result["text"].strip()
    print("\nTranscribed Text:")
    print(spoken_text)

    # Step C: sentiment prediction
    print("\nPredicting sentiment...")
    sentiment = predict_sentiment(spoken_text)
    print(f"Predicted Sentiment: {sentiment}")

    print("\nPipeline finished.")
    
    # Return both values for GUI integration
    return spoken_text, sentiment

# ===========================================================
#  TTS ENGINE (threaded)
# ===========================================================
# TTS lock for thread safety
_tts_lock = threading.Lock()

def speak_fallback(text):
    """Fallback TTS using Windows SAPI"""
    def _speak_fallback():
        try:
            # Use Windows PowerShell to speak
            cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"'
            subprocess.run(cmd, shell=True, capture_output=True)
            print(f"Fallback TTS completed for: {text}")
        except Exception as e:
            print(f"Fallback TTS Error: {e}")
    
    threading.Thread(target=_speak_fallback, daemon=True).start()

def speak(text):
    def _speak():
        with _tts_lock:  # Ensure only one TTS operation at a time
            try:
                print(f"Starting TTS for: {text}")
                
                # Force cleanup of any existing pyttsx3 instances
                gc.collect()
                
                # Try multiple initialization attempts
                engine = None
                for attempt in range(3):
                    try:
                        # Create a fresh engine
                        engine = pyttsx3.init(driverName='sapi5')  # Force SAPI5 on Windows
                        break
                    except Exception as e:
                        print(f"TTS init attempt {attempt + 1} failed: {e}")
                        time.sleep(0.2)
                
                if engine is None:
                    print("Failed to initialize TTS engine, using fallback")
                    speak_fallback(text)
                    return
                
                # Set properties
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 1.0)
                
                # Get and set voice
                voices = engine.getProperty('voices')
                if voices and len(voices) > 0:
                    engine.setProperty('voice', voices[0].id)
                
                # Clear any existing queue
                engine.stop()
                time.sleep(0.1)
                
                # Add text and speak
                engine.say(text)
                engine.runAndWait()
                
                print(f"TTS completed for: {text}")
                
            except Exception as e:
                print(f"TTS Error: {e}, trying fallback")
                import traceback
                traceback.print_exc()
                # Use fallback on error
                speak_fallback(text)
            finally:
                # Aggressive cleanup
                if engine is not None:
                    try:
                        engine.stop()
                        time.sleep(0.1)
                    except:
                        pass
                    try:
                        # Try to explicitly delete the engine
                        del engine
                    except:
                        pass
                
                # Force garbage collection
                gc.collect()
                time.sleep(0.1)

    # Run in a daemon thread so GUI stays responsive
    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()
    return thread

# ===========================================================
#  GUI APPLICATION
# ===========================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("🎙 Sentiment Analyzer")
app.geometry("800x650")

# ===========================================================
#  RESULT OUTPUT AREA
# ===========================================================
def update_result(sentiment):
    print(f"Updating result with sentiment: {sentiment}")
    result_label.configure(text=f"Result: {sentiment}")
    print(f"About to call speak with: 'The sentiment is {sentiment}'")
    speak(f"The sentiment is {sentiment}")
    print("Speak function called")

def update_speech_text(text):
    speech_text_box.configure(state="normal")
    speech_text_box.delete("0.0", "end")
    speech_text_box.insert("end", text)
    speech_text_box.configure(state="disabled")

# ===========================================================
#  SPEECH THREAD
# ===========================================================
def run_speech_thread():
    threading.Thread(target=_speech_mode, daemon=True).start()

def _speech_mode():
    try:
        # Indicate processing
        result_label.configure(text="Processing speech...")
        speech_text_box.configure(state="normal")
        speech_text_box.delete("0.0", "end")
        speech_text_box.insert("end", "Listening...")
        speech_text_box.configure(state="disabled")

        print("About to call run_pipeline()")
        # Run your pipeline: must return (transcribed_text, sentiment)
        speech_text, sentiment = run_pipeline()  
        print(f"Pipeline returned: text='{speech_text}', sentiment='{sentiment}'")

        update_speech_text(speech_text)
        update_result(sentiment)

    except Exception as e:
        print(f"Exception in _speech_mode: {e}")
        import traceback
        traceback.print_exc()
        result_label.configure(text="Error occurred!")
        speak("An error occurred during speech recognition.")

# ===========================================================
#  TEXT MODE
# ===========================================================
def run_text_prediction():
    text = text_box.get("0.0", "end").strip()
    if not text:
        result_label.configure(text="Please enter text.")
        speak("Please enter text.")
        return

    try:
        result_label.configure(text="Predicting sentiment...")
        sentiment = predict_sentiment(text)
        update_result(sentiment)
    except Exception as e:
        print(f"Exception in text prediction: {e}")
        result_label.configure(text="Error occurred!")
        speak("An error occurred during text analysis.")

# ===========================================================
#  UI LAYOUT
# ===========================================================
title = ctk.CTkLabel(
    app,
    text="🎙 Speech & Text Sentiment Analyzer",
    font=("Arial", 30, "bold")
)
title.pack(pady=20)

# --- SPEECH SECTION ---
speech_btn = ctk.CTkButton(
    app,
    text="🎤 Speak & Analyze",
    width=300,
    height=60,
    fg_color="#2ecc71",
    hover_color="#27ae60",
    corner_radius=15,
    command=run_speech_thread
)
speech_btn.pack(pady=15)

speech_label = ctk.CTkLabel(
    app,
    text="Recognized Speech:",
    font=("Arial", 18),
)
speech_label.pack()

speech_text_box = ctk.CTkTextbox(
    app,
    width=600,
    height=80,
    corner_radius=12,
    state="disabled",
    border_width=2
)
speech_text_box.pack(pady=5)

divider = ctk.CTkLabel(app, text="──────── OR ────────", font=("Arial", 15))
divider.pack(pady=10)

# --- TEXT INPUT SECTION ---
text_label = ctk.CTkLabel(
    app,
    text="Enter Text Manually:",
    font=("Arial", 18),
)
text_label.pack()

text_box = ctk.CTkTextbox(
    app,
    width=600,
    height=120,
    corner_radius=12,
    border_width=2
)
text_box.pack(pady=5)

predict_btn = ctk.CTkButton(
    app,
    text="Analyze Text",
    width=300,
    height=50,
    fg_color="#3498db",
    hover_color="#2980b9",
    corner_radius=15,
    command=run_text_prediction
)
predict_btn.pack(pady=15)

# --- RESULT FIELD ---
result_label = ctk.CTkLabel(
    app,
    text="Result:",
    font=("Arial", 24, "bold"),
    text_color="#f1c40f"
)
result_label.pack(pady=20)

def cleanup_and_exit():
    """Clean up and exit application"""
    app.quit()

# --- EXIT BUTTON ---
exit_btn = ctk.CTkButton(
    app,
    text="Exit",
    width=120,
    height=40,
    fg_color="#e74c3c",
    hover_color="#c0392b",
    corner_radius=10,
    command=cleanup_and_exit
)
exit_btn.pack(pady=10)

# ===========================================================
# MAIN EXECUTION
# ===========================================================
if __name__ == "__main__":
    print("Starting Sentiment Analyzer GUI...")
    print("All models loaded successfully!")
    print("GUI is ready to use.\n")
    app.mainloop()