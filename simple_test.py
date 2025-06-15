#!/usr/bin/env python3
"""
Simple test for Ollama setup
"""

import requests
import sys

def test_ollama_service():
    """Test if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            print(f"Ollama service is running!")
            print(f"Available models: {len(models)}")
            
            if models:
                print("Models found:")
                for model in models:
                    print(f"  - {model['name']}")
                return True
            else:
                print("No models downloaded yet.")
                print("You need to download a model first:")
                print("  ollama pull llama3.1:8b")
                print("  or")
                print("  ollama pull phi3:mini")
                return False
        else:
            print(f"Ollama service error: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Cannot connect to Ollama service.")
        print("Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print(f"Error testing Ollama: {e}")
        return False

def test_python_imports():
    """Test if required Python packages are available"""
    try:
        import ollama
        print("Ollama Python package: OK")
        return True
    except ImportError:
        print("Ollama Python package not found.")
        print("Install with: pip install ollama")
        return False

def main():
    print("=== XRec Ollama Setup Test ===")
    
    # Test Python imports
    print("\n1. Testing Python packages...")
    python_ok = test_python_imports()
    
    # Test Ollama service
    print("\n2. Testing Ollama service...")
    service_ok = test_ollama_service()
    
    print("\n=== Results ===")
    if python_ok and service_ok:
        print("SUCCESS: Setup is complete and ready to use!")
        print("\nYou can now run:")
        print("  python generation/item_profile/generate_profile.py")
        print("  python generation/user_profile/generate_profile.py")
        print("  python generation/explanation/generate_exp.py")
    elif python_ok and not service_ok:
        print("PARTIAL: Python packages OK, but need to download models")
        print("\nNext steps:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Download a model: ollama pull llama3.1:8b")
        print("3. Run this test again")
    else:
        print("FAILED: Setup incomplete")
        print("\nPlease check the error messages above")

if __name__ == "__main__":
    main()
