import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from transformers import ClapProcessor, ClapModel

# ========== CONFIG ==========
PROCESSED_AUDIO_DIR = "preprocessed_audio"
FSD50K_META_PATH = "fsd50k/FSD50K_ground_truth/eval.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ============================

# Initialize CLAP model
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(DEVICE)
model.eval()

# Load metadata
meta = pd.read_csv(FSD50K_META_PATH)
labels = sorted(meta['labels'].explode().unique())
print(f"Loaded {len(labels)} classes")

# Prepare text embeddings
prompts = [f"This is a sound of {label}" for label in labels]
text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
with torch.no_grad():
    text_embeds = model.get_text_features(**text_inputs).cpu()

def load_processed_audio(file_path):
    """Load and validate 48kHz/10s audio"""
    waveform, sr = torchaudio.load(file_path)
    assert sr == 48000, f"Invalid sample rate: {sr} (expected 48000)"
    assert waveform.shape[-1] == 480000, f"Invalid length: {waveform.shape[-1]} samples"
    return waveform.numpy().astype("float32")

# Initialize storage matrices
all_scores = np.zeros((len(meta), len(labels)))
all_targets = np.zeros((len(meta), len(labels)))

print("🔍 Evaluating preprocessed files...")
for idx, row in tqdm(meta.iterrows(), total=len(meta)):
    file_name = f"{row['fname']}.wav"
    file_path = os.path.join(PROCESSED_AUDIO_DIR, file_name)
    
    try:
        waveform = load_processed_audio(file_path)
    except Exception as e:
        print(f"⚠️ Error processing {file_name}: {str(e)}")
        continue
    
    # Process audio through CLAP
    inputs = processor(
        audios=[waveform],
        return_tensors="pt",
        sampling_rate=48000  # Must match preprocessing
    ).to(DEVICE)
    
    with torch.no_grad():
        audio_embed = model.get_audio_features(**inputs)
        sims = torch.nn.functional.cosine_similarity(
            audio_embed.cpu(),
            text_embeds,
            dim=-1
        )
    
    # Store predictions and targets
    all_scores[idx] = sims.numpy()
    for label in row['labels'].split(','):
        label = label.strip()
        if label in labels:
            all_targets[idx, labels.index(label)] = 1

# Calculate mean Average Precision
ap_scores = []
for class_idx in range(len(labels)):
    y_true = all_targets[:, class_idx]
    y_score = all_scores[:, class_idx]
    
    if np.sum(y_true) == 0:
        continue
    
    ap = average_precision_score(y_true, y_score)
    ap_scores.append(ap)

mean_ap = np.nanmean(ap_scores)
print(f"\n🎯 Final Zero-Shot mAP: {mean_ap:.4f} (Expected: ~0.3024)")