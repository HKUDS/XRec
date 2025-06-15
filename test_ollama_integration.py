#!/usr/bin/env python3
"""
Test script to verify Ollama integration works correctly
"""

import sys
import os
import torch
import numpy as np

# Add the explainer directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'explainer'))

from models.explainer import Explainer, OllamaClient, OllamaTokenizer
from utils.ollama_utils import ensure_ollama_ready, print_ollama_status


def test_ollama_client():
    """Test basic Ollama client functionality"""
    print("=== Testing Ollama Client ===")
    
    client = OllamaClient(model_name="llama3.1:8b")
    
    # Test simple generation
    prompt = "What is machine learning?"
    response = client.generate(prompt, max_tokens=50)
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("✅ Ollama client test passed!\n")
    
    return True


def test_ollama_tokenizer():
    """Test Ollama tokenizer functionality"""
    print("=== Testing Ollama Tokenizer ===")
    
    tokenizer = OllamaTokenizer()
    
    # Test tokenization
    text = "This is a test with <USER_EMBED> and <ITEM_EMBED> tokens."
    result = tokenizer(text)
    
    print(f"Input text: {text}")
    print(f"Tokenized: {result}")
    
    # Test batch tokenization
    batch_text = [
        "User profile: <USER_EMBED>",
        "Item description: <ITEM_EMBED>",
        "Explanation: <EXPLAIN_POS>"
    ]
    batch_result = tokenizer(batch_text)
    
    print(f"Batch input: {batch_text}")
    print(f"Batch tokenized: {batch_result}")
    print("✅ Ollama tokenizer test passed!\n")
    
    return True


def test_explainer_model():
    """Test the modified Explainer model"""
    print("=== Testing Explainer Model ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = Explainer(model_name="llama3.1:8b").to(device)
    
    # Create dummy embeddings
    batch_size = 2
    user_embed_size = 64
    item_embed_size = 64
    
    user_embedding = torch.randn(batch_size, user_embed_size).to(device)
    item_embedding = torch.randn(batch_size, item_embed_size).to(device)
    
    # Test input text
    input_text = [
        "User: <USER_EMBED> Item: <ITEM_EMBED> <EXPLAIN_POS>",
        "Recommend <ITEM_EMBED> to <USER_EMBED> because <EXPLAIN_POS>"
    ]
    
    print(f"Input text: {input_text}")
    
    # Test forward pass
    print("Testing forward pass...")
    input_ids, outputs, explain_pos = model.forward(user_embedding, item_embedding, input_text)
    print(f"Forward pass output shapes:")
    print(f"  Input IDs: {input_ids.shape}")
    print(f"  Outputs logits: {outputs.logits.shape}")
    print(f"  Explain position: {explain_pos.shape}")
    
    # Test loss calculation
    print("Testing loss calculation...")
    loss = model.loss(input_ids, outputs, explain_pos, device)
    print(f"Loss: {loss.item()}")
    
    # Test generation
    print("Testing generation...")
    generated_text = model.generate(user_embedding, item_embedding, input_text)
    print(f"Generated text: {generated_text}")
    
    print("✅ Explainer model test passed!\n")
    
    return True


def main():
    """Run all tests"""
    print("🚀 Testing Ollama Integration for XRec")
    print("=" * 50)
    
    # Check Ollama setup
    print("Checking Ollama setup...")
    if not ensure_ollama_ready("llama3.1:8b"):
        print("❌ Ollama setup failed. Please run 'python setup_ollama.py'")
        print_ollama_status()
        return 1
    
    print("✅ Ollama is ready!\n")
    
    try:
        # Run tests
        test_ollama_client()
        test_ollama_tokenizer()
        test_explainer_model()
        
        print("🎉 All tests passed!")
        print("The Ollama integration is working correctly.")
        print("\nYou can now run:")
        print("  python explainer/main.py --mode generate --dataset amazon --model_name llama3.1:8b")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
