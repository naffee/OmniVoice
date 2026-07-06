import os
import sys

print("Starting download of models...")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub is not installed. Please run inside a configured environment.")
    sys.exit(1)

models_to_download = [
    "k2-fsa/OmniVoice",
    "eustlb/higgs-audio-v2-tokenizer",
    "openai/whisper-large-v3-turbo"
]

for model_name in models_to_download:
    print(f"Downloading {model_name}...")
    try:
        snapshot_download(repo_id=model_name)
        print(f"Successfully downloaded {model_name}.")
    except Exception as e:
        print(f"Failed to download {model_name}: {e}")
        sys.exit(1)

print("All models pre-downloaded successfully!")
