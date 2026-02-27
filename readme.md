# Multilingual Neural Translator

A Bidirectional Seq2Seq LSTM model with Attention mechanism for translating between:

• English
• Hindi
• Punjabi

## Features

- Bidirectional Encoder
- Attention Mechanism
- GPU Training Support
- Flask Web Interface
- Beautiful UI
- Dynamic language selection

## Setup

1. Install dependencies
   pip install -r requirements.txt

2. Train model
   python train.py

3. Run Flask app
   python app.py

4. Open browser
   http://127.0.0.1:5000

## Dataset Format

source_lang,target_lang,source_text,target_text

## License

MIT