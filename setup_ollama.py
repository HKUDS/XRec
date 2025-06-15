#!/usr/bin/env python3
"""
Setup script for Ollama and required models for XRec project
This script will help you install Ollama and download the necessary models
"""

import subprocess
import sys
import os
import platform
import time

class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    END = "\033[0m"
    BOLD = "\033[1m"

def print_colored(message, color=Colors.GREEN):
    print(f"{color}{message}{Colors.END}")

def run_command(command, check=True):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=check)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_ollama_installed():
    """Check if Ollama is already installed"""
    success, _, _ = run_command("ollama --version", check=False)
    return success

def install_ollama():
    """Install Ollama based on the operating system"""
    system = platform.system().lower()
    
    print_colored("🚀 Installing Ollama...", Colors.BLUE)
    
    if system == "linux" or system == "darwin":  # Linux or macOS
        print_colored("Detected Unix-like system. Installing Ollama...", Colors.YELLOW)
        success, stdout, stderr = run_command("curl -fsSL https://ollama.ai/install.sh | sh")
        
        if success:
            print_colored("✅ Ollama installed successfully!", Colors.GREEN)
            return True
        else:
            print_colored(f"❌ Failed to install Ollama: {stderr}", Colors.RED)
            return False
            
    elif system == "windows":
        print_colored("🪟 Windows detected. Please install Ollama manually:", Colors.YELLOW)
        print_colored("1. Go to https://ollama.ai/download", Colors.YELLOW)
        print_colored("2. Download and run the Windows installer", Colors.YELLOW)
        print_colored("3. Run this script again after installation", Colors.YELLOW)
        return False
    else:
        print_colored(f"❌ Unsupported operating system: {system}", Colors.RED)
        return False

def start_ollama_service():
    """Start the Ollama service"""
    print_colored("🔄 Starting Ollama service...", Colors.BLUE)
    
    # Try to start Ollama in the background
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait a bit for the service to start
    time.sleep(3)
    
    # Check if the service is running
    success, _, _ = run_command("curl -s http://localhost:11434/api/tags", check=False)
    if success:
        print_colored("✅ Ollama service is running!", Colors.GREEN)
        return True
    else:
        print_colored("⚠️  Ollama service might not be running. You may need to start it manually with 'ollama serve'", Colors.YELLOW)
        return False

def pull_model(model_name):
    """Pull a specific model"""
    print_colored(f"📥 Downloading model: {model_name}", Colors.BLUE)
    print_colored("This may take a while depending on your internet connection...", Colors.YELLOW)
    
    success, stdout, stderr = run_command(f"ollama pull {model_name}")
    
    if success:
        print_colored(f"✅ Model {model_name} downloaded successfully!", Colors.GREEN)
        return True
    else:
        print_colored(f"❌ Failed to download model {model_name}: {stderr}", Colors.RED)
        return False

def list_available_models():
    """List locally available models"""
    success, stdout, stderr = run_command("ollama list", check=False)
    
    if success:
        print_colored("📋 Available models:", Colors.BLUE)
        print(stdout)
        return True
    else:
        print_colored("❌ Could not list models. Make sure Ollama is running.", Colors.RED)
        return False

def main():
    print_colored("🎯 XRec Ollama Setup Script", Colors.BOLD + Colors.BLUE)
    print_colored("=" * 50, Colors.BLUE)
    
    # Check if Ollama is installed
    if check_ollama_installed():
        print_colored("✅ Ollama is already installed!", Colors.GREEN)
    else:
        print_colored("📦 Ollama not found. Installing...", Colors.YELLOW)
        if not install_ollama():
            print_colored("❌ Installation failed. Please install Ollama manually.", Colors.RED)
            return
    
    # Start Ollama service
    start_ollama_service()
    
    # Recommended models for XRec
    recommended_models = [
        "llama3.1:8b",  # Primary recommendation
        "mistral:7b",   # Alternative option
    ]
    
    print_colored("\n🤖 Recommended models for XRec:", Colors.BLUE)
    for i, model in enumerate(recommended_models, 1):
        print_colored(f"{i}. {model}", Colors.YELLOW)
    
    print_colored("\nChoose an option:", Colors.BLUE)
    print_colored("1. Download llama3.1:8b (Recommended, ~4.7GB)", Colors.YELLOW)
    print_colored("2. Download mistral:7b (Alternative, ~4.1GB)", Colors.YELLOW)
    print_colored("3. Download both models", Colors.YELLOW)
    print_colored("4. Skip model download", Colors.YELLOW)
    print_colored("5. List currently available models", Colors.YELLOW)
    
    try:
        choice = input(f"{Colors.BLUE}Enter your choice (1-5): {Colors.END}")
        
        if choice == "1":
            pull_model("llama3.1:8b")
        elif choice == "2":
            pull_model("mistral:7b")
        elif choice == "3":
            for model in recommended_models:
                pull_model(model)
        elif choice == "4":
            print_colored("⏭️  Skipping model download.", Colors.YELLOW)
        elif choice == "5":
            list_available_models()
        else:
            print_colored("❌ Invalid choice.", Colors.RED)
            return
            
    except KeyboardInterrupt:
        print_colored("\n\n⏹️  Setup interrupted by user.", Colors.YELLOW)
        return
    
    print_colored("\n🎉 Setup complete!", Colors.GREEN)
    print_colored("You can now run the XRec generation scripts:", Colors.BLUE)
    print_colored("• python generation/item_profile/generate_profile.py", Colors.YELLOW)
    print_colored("• python generation/user_profile/generate_profile.py", Colors.YELLOW)
    print_colored("• python generation/explanation/generate_exp.py", Colors.YELLOW)
    
    print_colored("\n💡 Tips:", Colors.BLUE)
    print_colored("• If Ollama service stops, restart it with: ollama serve", Colors.YELLOW)
    print_colored("• To change models, edit MODEL_NAME in the generation scripts", Colors.YELLOW)
    print_colored("• Available models: ollama list", Colors.YELLOW)
    print_colored("• Pull new models: ollama pull <model_name>", Colors.YELLOW)

if __name__ == "__main__":
    main()

