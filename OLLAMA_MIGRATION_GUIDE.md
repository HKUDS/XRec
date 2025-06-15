# XRec Ollama Migration Guide

This document explains the changes made to migrate XRec from Hugging Face to Ollama for local, privacy-focused operation.

## 🎯 What Changed

### ✅ Benefits of the Migration
- **No Hugging Face Access Token Required**: Eliminates the need for HF authentication
- **Local Operation**: All processing happens on your machine
- **Privacy-First**: No data sent to external APIs
- **Cost-Free**: No API usage costs
- **Offline Capable**: Works without internet connection (after model download)
- **Easy Model Switching**: Simple parameter to change models

### 🔧 Technical Changes

#### 1. New Ollama Integration (`explainer/models/explainer.py`)
- **OllamaTokenizer**: Custom tokenizer wrapper for compatibility
- **OllamaClient**: Direct integration with Ollama API
- **Modified Explainer**: Redesigned to work with Ollama instead of HF models

#### 2. Enhanced Main Script (`explainer/main.py`)
- **Ollama Health Checks**: Automatic verification that Ollama is running
- **Model Parameter**: New `--model_name` argument for model selection
- **Better Error Handling**: Clear instructions when Ollama isn't set up

#### 3. Utility Functions (`explainer/utils/ollama_utils.py`)
- **Service Management**: Check and start Ollama service
- **Model Management**: Download and verify models
- **Status Reporting**: Detailed Ollama status information

#### 4. Updated Requirements (`requirements.txt`)
- **Reduced Dependencies**: Removed heavy HF packages
- **Essential Only**: Kept only necessary packages
- **Optional GPU**: CUDA packages are now optional

## 🚀 How to Use

### Quick Start
```bash
# 1. Install and setup Ollama
python setup_ollama.py

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run XRec with Ollama
python explainer/main.py --mode generate --dataset amazon --model_name llama3.1:8b
```

### Manual Setup
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Download model
ollama pull llama3.1:8b

# Run XRec
python explainer/main.py --mode generate --dataset amazon
```

### Model Options
```bash
# Use different models
python explainer/main.py --mode generate --dataset amazon --model_name mistral:7b
python explainer/main.py --mode generate --dataset amazon --model_name llama3.1:70b
```

## 🧪 Testing

Run the integration test to verify everything works:
```bash
python test_ollama_integration.py
```

This will test:
- Ollama service connectivity
- Model availability
- Tokenizer functionality
- Explainer model integration

## 🔍 Architecture Overview

### Before (Hugging Face)
```
Input → HF Tokenizer → LLaMA Model → MoE Adapters → Output
         ↓
    Requires HF Token & Internet
```

### After (Ollama)
```
Input → Custom Tokenizer → Ollama API → MoE Adapters → Output
         ↓
    Local Processing Only
```

### Key Components

1. **OllamaTokenizer**: Handles special tokens and maintains compatibility
2. **OllamaClient**: Communicates with local Ollama service
3. **MoE Adapters**: Preserved from original - convert embeddings to text features
4. **Explainer**: Modified to use Ollama while keeping training capabilities

## 🛠️ Troubleshooting

### Common Issues

#### "Ollama service not running"
```bash
# Start Ollama
ollama serve
```

#### "Model not found"
```bash
# Download the model
ollama pull llama3.1:8b
```

#### "Connection refused"
```bash
# Check if Ollama is running on correct port
curl http://localhost:11434/api/tags
```

#### "Import errors"
```bash
# Install dependencies
pip install -r requirements.txt
```

### Getting Help

1. **Check Ollama Status**: `python -c "from explainer.utils.ollama_utils import print_ollama_status; print_ollama_status()"`
2. **Run Test Script**: `python test_ollama_integration.py`
3. **Check Logs**: Look at Ollama service logs for detailed error information

## 📊 Performance Notes

- **Memory Usage**: Reduced due to removal of HF transformers
- **Startup Time**: Faster initialization (no model download from HF)
- **Generation Speed**: Comparable to HF, depends on local hardware
- **Model Size**: Same as before (models are equivalent)

## 🔄 Migration Checklist

- [x] Remove Hugging Face authentication
- [x] Implement Ollama client integration
- [x] Create custom tokenizer wrapper
- [x] Update main script with Ollama checks
- [x] Add utility functions for Ollama management
- [x] Update documentation and README
- [x] Create test script for verification
- [x] Simplify requirements.txt

## 🎉 Next Steps

1. **Test the Integration**: Run `python test_ollama_integration.py`
2. **Try Different Models**: Experiment with `mistral:7b`, `codellama:7b`, etc.
3. **Fine-tune Performance**: Adjust temperature and token limits as needed
4. **Scale Up**: Try larger models like `llama3.1:70b` if you have the resources

The migration maintains all the core functionality of XRec while providing a more accessible, privacy-focused, and cost-effective solution!
