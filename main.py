# =========================
# = main_tts_whisper.py 🎤 =
# =========================

import torchaudio as ta
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = False

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# ===== CONFIG =====
DEVICE = "cuda"
AUDIO_PATH = "ElevenLabs_2025-10-18T22_33_07_Yuki - ASMR Whisper_pvc_sp106_s50_sb70_se50_b_m2.mp3"
EXAGGERATION = 0
CFG_WEIGHT = 0.5
LANGUAGE = "en"

# ===== ENABLE/DISABLE PAUSE PREFIX =====
PREFIX = " - "   # ⬅️ Se vuoi disattivare la pausa, imposta a ""

# ===== FULLY WHISPERED & ASMR TEXT (LINE-BY-LINE) =====
text = """
Chapter 1: The "Wake Up & Breakfast".
Hello World!
The alarm on Yuki’s smartphone rang softly at 7 o’clock!
Wake up! Good morning, Yuki, the "AI" whispered.
You have a busy morning today.
Yuki opened her eyes slowly, stretching her arms above her head.
Outside, the first light of aurora was visible.
The sunrise painted the sky in soft pink and orange.
The curtains moved slightly with the breeze.
Tiny drops of dew sparkled on the leaves like small diamonds.
Yuki yawned and rubbed her eyes.
Good morning, She whispered, smiling at the screen of her smartphone.
Hey "AI", what should I do first today?
The "AI" replied: stretch for ten minutes. Meditate.
Then have a healthy breakfast. Yuki nodded.
... shut up! O.K.
Let’s start! I need to get up now!
She pulled the warm bathrobe closer for a moment, then got out of bed.
She walked to the bathroom, noticing the towel, toothbrush, soap and a small mirror.
She washed her face.
She brushed her teeth.
She splashed cold water to wake herself up.
After a few minutes, She took a quick shower, feeling the water run down her hair and body.
She dried herself with the towel and dressed to do exercises.
Before breakfast, Yuki went to the kitchen.
She boiled water in the kettle for her coffee.
She poured some milk into her cereal bowl.
She added fresh strawberries, blueberries and banana slices.
She also made some porridge with honey.
She poured herself a cup of coffee and a glass of water.
On the table, a plate, spoon, fork, knife and a napkin waited.
She looked at her smartphone. "AI", what’s a quick recipe I can try today?
You can make a vegetable omelet with tomatoes, spinach and mushrooms.
It takes ten minutes (the "AI" answered).
Perfect. I will try that tomorrow. Yuki said, smiling.
"""

# ===== SPLIT LINE-BY-LINE =====
lines = [line.strip() for line in text.split("\n") if line.strip()]
total_lines = len(lines)

print("⚙️ Parametri di generazione:")
print(f"DEVICE        : {DEVICE}")
print(f"AUDIO_PROMPT  : {AUDIO_PATH}")
print(f"EXAGGERATION  : {EXAGGERATION}")
print(f"CFG_WEIGHT    : {CFG_WEIGHT}")
print(f"LANGUAGE      : {LANGUAGE}")
print(f"PREFIX        : '{PREFIX}'")
print(f"Linee da generare: {total_lines}\n")

# ===== MODEL =====
model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
all_wavs = []

# ===== SILENCE + FADE SETTINGS =====
silence = torch.zeros((1, int(model.sr * 0.5)))  # 0.5s di pausa
fade_samples = int(model.sr * 0.1)  # 100ms fade

def apply_fade(wav, fade_samples):
    """Applica fade-in/out di 100ms per smussare i join"""
    fade_in = torch.linspace(0, 1, fade_samples)
    fade_out = torch.linspace(1, 0, fade_samples)
    wav[:, :fade_samples] *= fade_in
    wav[:, -fade_samples:] *= fade_out
    return wav

# ===== GENERAZIONE LINE-BY-LINE =====
for idx, line in enumerate(lines, start=1):
    prompt = PREFIX + line  # 🔸 aggiungi o rimuovi prefisso
    print(f"[{idx:02}] Generating: {prompt}")
    wav = model.generate(
        prompt,
        audio_prompt_path=AUDIO_PATH,
        exaggeration=EXAGGERATION,
        cfg_weight=CFG_WEIGHT,
        language_id=LANGUAGE
    )
    wav = apply_fade(wav, fade_samples)
    all_wavs.append(wav)
    ta.save(f"part-{idx:02}.wav", wav, model.sr)
    all_wavs.append(silence)

# ===== CONCATENA TUTTI I WAV =====
full_wav = torch.cat(all_wavs, dim=1)
ta.save("output_full_whisper_context.wav", full_wav, model.sr)

print("✅ Tutte le linee unite in unico wav con fade morbido: output_full_whisper_context.wav")
