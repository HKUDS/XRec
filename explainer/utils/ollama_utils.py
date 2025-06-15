"""
Utility functions for Ollama integration
"""

import requests
import subprocess
import time
import sys


def check_ollama_running(base_url="http://localhost:11434"):
    """Check if Ollama service is running"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def check_model_available(model_name="llama3.1:8b", base_url="http://localhost:11434"):
    """Check if a specific model is available in Ollama"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]
            return model_name in available_models
        return False
    except requests.exceptions.RequestException:
        return False


def start_ollama_service():
    """Attempt to start Ollama service"""
    try:
        # Try to start Ollama in the background
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait a bit for the service to start
        time.sleep(3)
        
        # Check if it's running
        return check_ollama_running()
    except FileNotFoundError:
        return False


def pull_model(model_name="llama3.1:8b"):
    """Pull a model if it's not available"""
    try:
        result = subprocess.run(["ollama", "pull", model_name], 
                              capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def ensure_ollama_ready(model_name="llama3.1:8b"):
    """Ensure Ollama is running and the model is available"""
    print("Checking Ollama setup...")
    
    # Check if Ollama is running
    if not check_ollama_running():
        print("Ollama service is not running. Attempting to start...")
        if not start_ollama_service():
            print("❌ Failed to start Ollama service.")
            print("Please install and start Ollama manually:")
            print("1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
            print("2. Start service: ollama serve")
            print("3. Or run: python setup_ollama.py")
            return False
        print("✅ Ollama service started successfully!")
    else:
        print("✅ Ollama service is running!")
    
    # Check if model is available
    if not check_model_available(model_name):
        print(f"Model {model_name} is not available. Attempting to download...")
        if not pull_model(model_name):
            print(f"❌ Failed to download model {model_name}.")
            print(f"Please download manually: ollama pull {model_name}")
            print("Or run: python setup_ollama.py")
            return False
        print(f"✅ Model {model_name} downloaded successfully!")
    else:
        print(f"✅ Model {model_name} is available!")
    
    return True


def get_available_models(base_url="http://localhost:11434"):
    """Get list of available models"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except requests.exceptions.RequestException:
        return []


def print_ollama_status():
    """Print detailed Ollama status"""
    print("=== Ollama Status ===")
    
    if check_ollama_running():
        print("✅ Ollama service: Running")
        models = get_available_models()
        if models:
            print("✅ Available models:")
            for model in models:
                print(f"   - {model}")
        else:
            print("⚠️  No models available")
    else:
        print("❌ Ollama service: Not running")
        print("To start Ollama:")
        print("   ollama serve")
    
    print("==================")
