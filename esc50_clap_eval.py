import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from transformers import ClapProcessor, ClapModel
from sklearn.metrics import accuracy_score
from torch.nn.functional import cosine_similarity

# ========== CONFIG ==========
ESC50_AUDIO_DIR = "ESC-50-master/ESC-50-master/audio"
ESC50_META_PATH = "ESC-50-master/ESC-50-master/meta/esc50.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ============================

print("🧠 Loading model and processor...")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(DEVICE)
print("✅ Model loaded.")

# Load ESC-50 metadata
meta = pd.read_csv(ESC50_META_PATH)
labels = sorted(meta['category'].unique())
prompts = [f"This is a sound of {label}" for label in labels]

# Get text embeddings
print("📝 Encoding text prompts...")
text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
with torch.no_grad():
    text_embeds = model.get_text_features(**text_inputs)

# Audio preprocessing
# def load_audio(file_path):
#     waveform, sr = torchaudio.load(file_path)
#     if sr != 48000:
#         resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
#         waveform = resampler(waveform)
#     return waveform

def load_audio(file_path):
    waveform, sr = torchaudio.load(file_path)

    # Convert to mono
    waveform = waveform.mean(dim=0)

    # Resample to 48kHz if needed
    if sr != 48000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
        waveform = resampler(waveform)

    # Trim or pad to exactly 10 seconds (480,000 samples)
    desired_len = 480000
    if waveform.shape[0] < desired_len:
        pad_len = desired_len - waveform.shape[0]
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
    else:
        waveform = waveform[:desired_len]

    # 🔥 Convert to NumPy float32 (this is the key!)
    waveform = waveform.numpy().astype("float32")
    return waveform


# Inference loop
true_labels = []
predicted_labels = []

print("🔍 Running inference on ESC-50...")
for _, row in tqdm(meta.iterrows(), total=len(meta)):
    file_path = os.path.join(ESC50_AUDIO_DIR, row['filename'])
    waveform = load_audio(file_path)

    # inputs = processor(audios=waveform, return_tensors="pt", sampling_rate=48000).to(DEVICE)
    inputs = processor(audios=[waveform], return_tensors="pt", sampling_rate=48000).to(DEVICE)


    with torch.no_grad():
        audio_embed = model.get_audio_features(**inputs)
        sims = cosine_similarity(audio_embed, text_embeds)
        pred_label = labels[sims.argmax().item()]

    true_labels.append(row['category'])
    predicted_labels.append(pred_label)

# Accuracy
acc = accuracy_score(true_labels, predicted_labels)
print(f"\n🎯 Zero-Shot Accuracy on ESC-50: {acc * 100:.2f}%")
