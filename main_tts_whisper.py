# ==========================
# = main_tts_whisper.py 🎤 =
# Edoardo
# 20 Ottobre 2025
# ==========================

import os
import shutil
import torchaudio as ta
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = False

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# ===== CONFIG =====
DEVICE = "cuda"
AUDIO_PATH = "ElevenLabs_2025-10-18T22_33_07_Yuki - ASMR Whisper_pvc_sp106_s50_sb70_se50_b_m2.mp3"
EXAGGERATION = 0
CFG_WEIGHT = 0.12
LANGUAGE = "en"
PREFIX = " - "

INPUT_FILE = "input.txt"
TMP_DIR = "tmpwav"
OUTPUT_WAV = "output_full_whisper_context.wav"
OUTPUT_MP3 = "output_full_whisper_context.mp3"

# ===== LOAD TEXT =====
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ Il file di input '{INPUT_FILE}' non esiste!")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

lines = [line.strip() for line in text.split("\n") if line.strip()]
total_lines = len(lines)

print("⚙️ Parametri di generazione:")
print(f"DEVICE        : {DEVICE}")
print(f"AUDIO_PROMPT  : {AUDIO_PATH}")
print(f"EXAGGERATION  : {EXAGGERATION}")
print(f"CFG_WEIGHT    : {CFG_WEIGHT}")
print(f"LANGUAGE      : {LANGUAGE}")
print(f"PREFIX        : '{PREFIX}'")
print(f"Linee da generare: {total_lines}")
print(f"Input file    : {INPUT_FILE}\n")

# ===== TMP FOLDER =====
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
os.makedirs(TMP_DIR)

# ===== MODEL =====
model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
all_wavs = []

# ===== SILENCE + FADE =====
silence = torch.zeros((1, int(model.sr * 0.2)))
fade_samples = int(model.sr * 0.1)

def apply_fade(wav, fade_samples):
    fade_in = torch.linspace(0, 1, fade_samples)
    fade_out = torch.linspace(1, 0, fade_samples)
    wav[:, :fade_samples] *= fade_in
    wav[:, -fade_samples:] *= fade_out
    return wav

# ===== GENERAZIONE LINE-BY-LINE =====
for idx, line in enumerate(lines, start=1):
    prompt = PREFIX + line
    print(f"[{idx:02}] Generating: {prompt}")
    wav = model.generate(
        prompt,
        audio_prompt_path=AUDIO_PATH,
        exaggeration=EXAGGERATION,
        cfg_weight=CFG_WEIGHT,
        language_id=LANGUAGE,
    )
    wav = apply_fade(wav, fade_samples)
    all_wavs.append(wav)
    part_path = os.path.join(TMP_DIR, f"part-{idx:02}.wav")
    ta.save(part_path, wav, model.sr)
    all_wavs.append(silence)

# ===== CONCATENA =====
full_wav = torch.cat(all_wavs, dim=1)
ta.save(OUTPUT_WAV, full_wav, model.sr)
ta.save(OUTPUT_MP3, full_wav, model.sr, format="mp3")

# ===== CLEANUP =====
shutil.rmtree(TMP_DIR)

print(f"\n✅ Tutte le linee unite e salvate:")
print(f" - WAV: {OUTPUT_WAV}")
print(f" - MP3: {OUTPUT_MP3}")
print(f"🧹 Cartella temporanea '{TMP_DIR}' rimossa.")
