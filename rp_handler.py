import io
import os
import base64
import requests
import soundfile as sf
import numpy as np
import torch
import runpod
from omnivoice import OmniVoice

# Initialize variables
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None

def get_model():
    global model
    if model is None:
        print("Loading OmniVoice model...")
        # We load ASR model too, so voice cloning without ref_text works seamlessly
        model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map=device,
            dtype=torch.float16,
            load_asr=True
        )
        print("OmniVoice model loaded successfully.")
    return model

def handler(event):
    """
    RunPod Serverless Handler for OmniVoice.
    
    Expected JSON Input:
    {
      "input": {
        "text": "Hello, this is a test of OmniVoice on RunPod Serverless.",
        "mode": "auto", // "auto", "voice_cloning", "voice_design"
        
        // Voice Cloning parameters
        "ref_audio": "base64_string_or_http_url",
        "ref_text": "Optional transcription of the reference audio.",
        
        // Voice Design parameters
        "instruct": "female, low pitch, british accent",
        
        // General generation configuration (passed to generate method)
        "language": "en",
        "num_step": 32,
        "speed": 1.0,
        "duration": null,
        "guidance_scale": 2.0,
        "postprocess_output": true
      }
    }
    """
    params = event.get("input", {})
    text = params.get("text")
    if not text:
        return {"error": "Missing required field 'text'"}

    mode = params.get("mode", "auto")
    ref_audio_input = params.get("ref_audio")
    ref_text = params.get("ref_text")
    instruct = params.get("instruct")
    
    # Extract general generation config overrides
    gen_config = {}
    for key in ["language", "num_step", "speed", "duration", "guidance_scale", "postprocess_output"]:
        if key in params:
            gen_config[key] = params[key]
            
    # Load model
    try:
        model_instance = get_model()
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}

    temp_ref_file = None
    try:
        # 1. Voice Cloning Mode
        if mode == "voice_cloning" or (ref_audio_input and mode != "voice_design"):
            if not ref_audio_input:
                return {"error": "ref_audio parameter is required for voice cloning mode"}
            
            temp_ref_file = "temp_ref_audio.wav"
            
            # Download URL or decode base64
            if ref_audio_input.startswith("http://") or ref_audio_input.startswith("https://"):
                print(f"Downloading reference audio from URL: {ref_audio_input}")
                r = requests.get(ref_audio_input, timeout=30)
                r.raise_for_status()
                with open(temp_ref_file, "wb") as f:
                    f.write(r.content)
            else:
                print("Decoding reference audio from base64")
                if "," in ref_audio_input:
                    ref_audio_input = ref_audio_input.split(",")[1]
                audio_data = base64.b64decode(ref_audio_input)
                with open(temp_ref_file, "wb") as f:
                    f.write(audio_data)
            
            print(f"Generating audio with Voice Cloning. ref_text: '{ref_text}'")
            audio = model_instance.generate(
                text=text,
                ref_audio=temp_ref_file,
                ref_text=ref_text,
                **gen_config
            )
            
        # 2. Voice Design Mode
        elif mode == "voice_design" or (instruct and mode != "voice_cloning"):
            if not instruct:
                return {"error": "instruct parameter is required for voice design mode"}
            
            print(f"Generating audio with Voice Design. instruct: '{instruct}'")
            audio = model_instance.generate(
                text=text,
                instruct=instruct,
                **gen_config
            )
            
        # 3. Auto Mode
        else:
            print("Generating audio with Auto voice style")
            audio = model_instance.generate(
                text=text,
                **gen_config
            )
        
        # Determine sampling rate
        sr = model_instance.sampling_rate if model_instance.sampling_rate is not None else 24000
        
        # Package and return output
        output_buffer = io.BytesIO()
        sf.write(output_buffer, audio[0], sr, format='WAV')
        base64_audio = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        
        return {
            "audio": base64_audio,
            "sampling_rate": sr,
            "format": "wav"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Inference execution failed: {str(e)}"}
        
    finally:
        # Cleanup temp file
        if temp_ref_file and os.path.exists(temp_ref_file):
            try:
                os.remove(temp_ref_file)
            except Exception:
                pass

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
