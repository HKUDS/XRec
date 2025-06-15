#!/usr/bin/env python3
"""
Quick check script to verify the Ollama migration was successful
"""

import sys
import os
import importlib.util

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (NOT FOUND)")
        return False

def check_import(module_path, description):
    """Check if a module can be imported"""
    try:
        spec = importlib.util.spec_from_file_location("test_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"✅ {description}: Can import successfully")
        return True
    except Exception as e:
        print(f"❌ {description}: Import failed - {e}")
        return False

def check_ollama_classes():
    """Check if Ollama classes are properly defined"""
    try:
        sys.path.append('explainer')
        from models.explainer import OllamaTokenizer, OllamaClient, Explainer
        
        # Test basic instantiation
        tokenizer = OllamaTokenizer()
        client = OllamaClient()
        
        print("✅ Ollama classes: All classes can be instantiated")
        return True
    except Exception as e:
        print(f"❌ Ollama classes: Failed to instantiate - {e}")
        return False

def main():
    """Run migration verification checks"""
    print("🔍 XRec Ollama Migration Verification")
    print("=" * 50)
    
    checks_passed = 0
    total_checks = 0
    
    # Check core files
    files_to_check = [
        ("explainer/models/explainer.py", "Modified Explainer model"),
        ("explainer/utils/ollama_utils.py", "Ollama utilities"),
        ("explainer/main.py", "Updated main script"),
        ("setup_ollama.py", "Ollama setup script"),
        ("test_ollama_integration.py", "Integration test script"),
        ("OLLAMA_MIGRATION_GUIDE.md", "Migration guide"),
        ("requirements.txt", "Updated requirements")
    ]
    
    print("\n📁 File Checks:")
    for filepath, description in files_to_check:
        if check_file_exists(filepath, description):
            checks_passed += 1
        total_checks += 1
    
    # Check imports
    print("\n📦 Import Checks:")
    import_checks = [
        ("explainer/models/explainer.py", "Explainer module"),
        ("explainer/utils/ollama_utils.py", "Ollama utilities"),
        ("explainer/main.py", "Main script")
    ]
    
    for filepath, description in import_checks:
        if os.path.exists(filepath) and check_import(filepath, description):
            checks_passed += 1
        total_checks += 1
    
    # Check Ollama classes
    print("\n🤖 Ollama Classes Check:")
    if check_ollama_classes():
        checks_passed += 1
    total_checks += 1
    
    # Check for removed HF dependencies
    print("\n🚫 Hugging Face Dependency Check:")
    try:
        with open("explainer/models/explainer.py", "r") as f:
            content = f.read()
            if "huggingface_hub" not in content and "from_pretrained" not in content:
                print("✅ Hugging Face dependencies: Successfully removed")
                checks_passed += 1
            else:
                print("❌ Hugging Face dependencies: Still present in code")
        total_checks += 1
    except Exception as e:
        print(f"❌ Could not check HF dependencies: {e}")
        total_checks += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Migration Check Results: {checks_passed}/{total_checks} passed")
    
    if checks_passed == total_checks:
        print("🎉 Migration appears to be successful!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Setup Ollama: python setup_ollama.py")
        print("3. Test integration: python test_ollama_integration.py")
        print("4. Run XRec: python explainer/main.py --mode generate --dataset amazon")
        return 0
    else:
        print("⚠️  Some checks failed. Please review the issues above.")
        print("\nFor help, see: OLLAMA_MIGRATION_GUIDE.md")
        return 1

if __name__ == "__main__":
    exit(main())
