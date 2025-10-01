

```markdown
# AI Pulse 

AI Pulse is a lightweight intent-based chatbot using PyTorch and NLTK. It classifies user messages into predefined intents and responds with dynamic answers or triggers custom functions like fetching stock information.

## Features

- **Intent Classification**: Uses neural network to classify user messages into predefined intents
- **Custom Function Mapping**: Triggers Python functions based on specific intents
- **Natural Language Processing**: Leverages NLTK for tokenization and lemmatization
- **PyTorch Backend**: Neural network model for accurate intent prediction
- **JSON Configuration**: Easy-to-edit intents and responses in JSON format
- **Model Persistence**: Save and load trained models for faster startup

## Project Structure

```
AI-Pulse/

├── main.py                        # Main chatbot application

├── intents.json                   # Intent definitions and training data

├── chatbot_model.pth              # Trained PyTorch model (generated)

├── dimensions.json                # Model vocabulary and dimensions (generated)

└── README.md                      # This file
```

## Installation

### Step 1: Install Dependencies

```bash
pip install torch nltk numpy
```

### Step 2: Verify Installation

```bash
python --version
pip list | grep -E "(torch|nltk|numpy)"
```

## Usage

### Running the Chatbot

1. **First Time Setup** (Train a new model):
   - Uncomment the training section in `main.py`:
   ```python
   # TRAIN NEW MODEL (uncomment this block to retrain)
   print("=== TRAINING NEW MODEL ===")
   assistant = ChatbotAssistant(INTENTS_FILE, {'stocks': get_stocks})
   assistant.parse_intents()
   assistant.prepare_data()
   assistant.train_model(epochs=300)
   assistant.save_model(MODEL_FILE, DIMS_FILE)
   ```

2. **Run the application**:
   ```bash
   python main.py
   ```

3. **Chat with AI Pulse**:
   ```
   AI Pulse is ready! Type '/quit' to exit.
   You: Hello
   AI Pulse: Hello! I'm AI Pulse, how can I help you today?
   You: What's your name?
   AI Pulse: I'm AI Pulse, your friendly chatbot assistant!
   You: What are my stocks?
   AI Pulse: Here are your stocks!
   Your stocks: AAPL, META, NVDA
   You: /quit
   ```

### Using Pre-trained Model

After initial training, comment the training section and uncomment the loading section:

```python
# LOAD EXISTING MODEL
print("=== LOADING EXISTING MODEL ===")
assistant = ChatbotAssistant(INTENTS_FILE, {'stocks': get_stocks})
assistant.parse_intents()
assistant.load_model(MODEL_FILE, DIMS_FILE)
```

## Customization

### Adding New Intents

Edit `intents.json` to add new conversation patterns:

```json
{
    "tag": "weather",
    "patterns": [
        "What's the weather like?",
        "How's the weather today?",
        "Is it raining?"
    ],
    "responses": [
        "I don't have weather data right now.",
        "You might want to check a weather app for that!"
    ]
}
```

### Adding Custom Functions

1. Define your function in `main.py`:
```python
def get_weather():
    print("Weather function triggered!")
```

2. Map it to an intent in the ChatbotAssistant:
```python
assistant = ChatbotAssistant(INTENTS_FILE, {
    'stocks': get_stocks,
    'weather': get_weather
})
```

## Configuration

### Intent File Structure

The `intents.json` file contains:
- `tag`: Unique identifier for the intent
- `patterns`: User message examples for training
- `responses`: Possible bot responses

### Model Configuration

- **Network Architecture**: 3-layer neural network (128-64-output)
- **Activation**: ReLU with Dropout (0.5) for regularization
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: CrossEntropyLoss

## Dependencies

- **torch**: Deep learning framework for the neural network
- **nltk**: Natural language processing for text tokenization
- **numpy**: Numerical computations and array handling

## Troubleshooting

### Common Issues

1. **NLTK Data Missing**:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
   ```

2. **Model Not Found**:
   - Train the model first by uncommenting the training section

3. **Import Errors**:
   - Ensure all dependencies are installed: `pip install torch nltk numpy`

### Performance Tips

- Add more training patterns for better accuracy
- Increase epochs for better training (300+ recommended)
- Balance the number of patterns per intent
- Use diverse and natural language patterns

## Development

### Training Process

1. **Data Preparation**: Tokenizes and lemmatizes patterns
2. **Vocabulary Building**: Creates bag-of-words representation
3. **Model Training**: Trains neural network on intent classification
4. **Model Saving**: Saves trained model and vocabulary for future use

### Architecture

- **ChatbotModel**: PyTorch neural network classifier
- **ChatbotAssistant**: Main class handling training and inference
- **Bag-of-Words**: Simple but effective text representation
- **Intent-Response Mapping**: Direct mapping between classified intents and responses



---
