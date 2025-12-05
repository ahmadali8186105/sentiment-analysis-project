# 🎙️ Sentiment Analysis with Speech Recognition

A complete machine learning pipeline for sentiment analysis with speech-to-text capabilities and an interactive GUI.


## 👥 Team

- **BITF22M036**: zara zainab
- **BITF22M038**: ahmad ali


## ✨ Features

- 🎤 **Speech-to-Text**: Record audio and convert to text using Whisper AI
- 🧠 **Advanced ML Pipeline**: Enhanced sentiment analysis with 27+ statistical features
- 💬 **Text-to-Speech**: Audio feedback for predictions
- 🖥️ **Modern GUI**: Dark-themed interface built with CustomTkinter
- 📊 **Feature Engineering**: TF-IDF + statistical text features for better accuracy
- 🔄 **Dual Input**: Supports both speech input and manual text entry

## 🖥️ System Requirements

- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: At least 2GB free space
- **Microphone**: Required for speech input feature
- **Git**: For cloning the repository


## 📁 Project Structure

```
sentiment-analysis-project/
├── complete_pipeline.py          # Complete training pipeline
├── sentiment_analyzer_app.py     # GUI application
├── README.md                     # This file
├── requirements.txt              # Dependencies list
├── data/                         # Data files (parquet files)
├── models/                       # Generated model files
│   ├── sentiment_lr_model_enhanced.pkl
│   ├── sentiment_lr_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   ├── feature_info.pkl
│   └── feature_scaler.pkl
└── temp/                         # Temporary audio files
```


## 📥 Step 1: Clone Repository

```bash
   git clone https://github.com/ahmadali8186105/sentiment-analysis-project.git
   cd sentiment-analysis-project
```

## 🐍 Step 2: Create Environment

### For Windows:

**Create virtual environment**:
   ```bash
   python -m venv sentiment_env
   ```

**Activate the environment**:
   ```bash
   # For Command Prompt:
   sentiment_env\Scripts\activate
   
   # For PowerShell:
   sentiment_env\Scripts\Activate.ps1
   ```

### For Mac/Linux:

**Create virtual environment**:
   ```bash
   python3 -m venv sentiment_env
   ```
**Activate the environment**:
   ```bash
   source sentiment_env/bin/activate
   ```

## 📦 Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## ▶️ Step 3: How to Run

### First Time Setup (Train the Model):

```bash
python complete_pipeline.py
python sentiment_analyzer_app.py
```

### For Future Use:
1. **Activate environment**: `sentiment_env\Scripts\activate` (Windows) or `source sentiment_env/bin/activate` (Mac/Linux)
2. **Run app**: `python sentiment_analyzer_app.py`

## 🖱️ Step 4: How to Use the GUI

### Method 1: Speech Analysis

1. **Click the "🎤 Speak & Analyze" button**
2. **record a clear speech of 5 seconds**:
   
### Method 2: Text Analysis

1. **Type Your Text**:
2. **Click "Analyze Text"**:
   
### Understanding Results

- **🟢 Positive**: Happy, excited, satisfied, joyful emotions
- **🔴 Negative**: Sad, angry, frustrated, disappointed emotions
  
### Example Usage Scenarios

#### Happy Scenarios:
- "I just got promoted at work and I'm so excited!"
- "This restaurant has the best food I've ever tasted"
- "I love spending time with my family on weekends"

#### Sad Scenarios:
- "I'm really disappointed with this purchase"
- "The movie was boring and I wasted my time"
- "I'm feeling stressed about my exams"

## 🛠️ Troubleshooting

### Common Issues and Solutions:

#### 1. "Module not found" errors
```bash
# Solution: Install the missing module
pip install <module-name>
```

#### 2. "No audio input device found"
- **Windows**: Check microphone permissions in Settings > Privacy > Microphone
- **Mac**: System Preferences > Security & Privacy > Microphone
- **Linux**: Check `alsamixer` settings

#### 3. "Whisper model download fails"
- Ensure stable internet connection
- The model will download automatically on first run (may take a few minutes)

#### 4. "TTS not working"
- **Windows**: Should work out of the box
- **Mac**: `brew install espeak`
- **Linux**: `sudo apt-get install espeak espeak-data`

#### 5. "Feature mismatch error"
- Delete old model files and retrain:
  ```bash
  rm *.pkl
  python complete_pipeline.py
  ```


