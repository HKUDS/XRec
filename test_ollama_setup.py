#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify Ollama setup for XRec project
"""

import sys
import os

# Add the generation utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'generation'))

try:
    from utils.ollama_client import get_ollama_response, OllamaClient
    print("Successfully imported Ollama client")
except ImportError as e:
    print(f"Failed to import Ollama client: {e}")
    sys.exit(1)

def test_ollama_connection():
    """Test basic Ollama connection"""
    print("\n🔍 Testing Ollama connection...")

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Connected to Ollama service")
            print(f"📋 Available models: {len(models)}")
            if models:
                for model in models:
                    print(f"   - {model['name']}")
                return True
            else:
                print("⚠️  No models downloaded yet")
                return False
        else:
            print(f"❌ Ollama service responded with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {e}")
        print("💡 Make sure Ollama is running with: ollama serve")
        return False

def test_simple_generation():
    """Test simple text generation"""
    print("\n🧪 Testing text generation...")
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Respond with exactly one word."},
            {"role": "user", "content": "Say 'Hello'"}
        ]
        
        response = get_ollama_response(messages, temperature=0.1)
        print(f"✅ Generated response: '{response.strip()}'")
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate text: {e}")
        return False

def test_xrec_style_generation():
    """Test XRec-style profile generation"""
    print("\n🎯 Testing XRec-style generation...")
    
    try:
        # Test item profile generation
        system_prompt = """You will serve as an assistant to help me summarize which types of users would enjoy a specific business.
I will provide you with the basic information of that business.
Please provide your answer in JSON format in one line, following this structure:
{   
    "summarization": "A summarization of what types of users would enjoy this business"
}
Please ensure that the "summarization" is no longer than 50 words."""

        user_prompt = """{
    "name": "Test Coffee Shop",
    "city": "San Francisco",
    "categories": "Coffee, Cafe, Breakfast"
}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = get_ollama_response(messages, temperature=0.7)
        print(f"✅ XRec-style response generated:")
        print(f"   {response.strip()}")
        return True
        
    except Exception as e:
        print(f"❌ Failed XRec-style generation: {e}")
        return False

def main():
    print("🎯 XRec Ollama Setup Test")
    print("=" * 40)
    
    tests = [
        ("Connection Test", test_ollama_connection),
        ("Simple Generation", test_simple_generation),
        ("XRec-style Generation", test_xrec_style_generation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"⚠️  {test_name} failed")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Ollama setup is working correctly.")
        print("\n🚀 You can now run the XRec generation scripts:")
        print("   • python generation/item_profile/generate_profile.py")
        print("   • python generation/user_profile/generate_profile.py")
        print("   • python generation/explanation/generate_exp.py")
    else:
        print("❌ Some tests failed. Please check your Ollama setup.")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure Ollama is installed: ollama --version")
        print("   2. Start Ollama service: ollama serve")
        print("   3. Download a model: ollama pull llama3.1:8b")
        print("   4. Check available models: ollama list")

if __name__ == "__main__":
    main()
