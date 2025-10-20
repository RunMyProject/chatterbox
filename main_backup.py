# ===================
# = main.py 🎤  =====
# ===================

import torchaudio as ta
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = False

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import sox  # per rallentamento opzionale
from faster_whisper import WhisperModel  # trascrizione opzionale

# ===== CONFIG =====
DEVICE = "cuda"
AUDIO_PATH = "ElevenLabs_2025-10-18T22_33_07_Yuki - ASMR Whisper_pvc_sp106_s50_sb70_se50_b_m2.mp3"
EXAGGERATION = 0.25
CFG_WEIGHT = 0.5
LANGUAGE = "en"
SLOW = False
SLOW_RATE = 0.8
WHISPER = True

# ===== TESTO TRADOTTO E FORMATTATO =====
text = """
Chapter 1a – The Morning Light (Wake Up & Breakfast)

Hello World! 
The alarm on Yuki’s smartphone rang softly at seven o’clock. 
“Wake up! Good morning, Yuki!” the AI whispered. 
“You have a busy morning today!” 
Yuki opened her eyes slowly, stretching her arms above her head. 
Outside, the first light of aurora was visible. 
The sunrise painted the sky in soft pink and orange. 
The curtains moved slightly with the breeze. 
Tiny drops of dew sparkled on the leaves like small diamonds.

Yuki yawned and rubbed her eyes. 
“Good morning,” she whispered, smiling at the screen of her smartphone. 
“AI, what should I do first today?” 
The AI replied: “Stretch for ten minutes. Meditate. Then have a healthy breakfast.” 
Yuki nodded. “Okay. Let’s start. I need to get up now.”

She pulled the warm bathrobe closer for a moment, then got out of bed. 
She walked to the bathroom, noticing the towel, toothbrush, soap, and a small mirror. 
She washed her face. 
She brushed her teeth. 
She splashed cold water to wake herself up. 
After a few minutes, she took a quick shower, feeling the water run down her hair and body. 
She dried herself with the towel and dressed to do exercises.

Before breakfast, Yuki went to the kitchen. 
She boiled water in the kettle for her coffee. 
She poured some milk into her cereal bowl. 
She added fresh strawberries, blueberries, and banana slices. 
She also made some porridge with honey. 
She poured herself a cup of coffee and a glass of water. 
On the table, a plate, spoon, fork, knife, and a napkin waited.

She looked at her smartphone. 
“AI, what’s a quick recipe I can try today?” 
“You can make a vegetable omelet with tomatoes, spinach, and mushrooms. 
It takes ten minutes,” the AI answered.
“Perfect. I will try that tomorrow,” Yuki said, smiling.
"""

# ===== SUDDIVISIONE IN PARAGRAFI =====
paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
total_paragraphs = len(paragraphs)

# ===== STAMPA PARAMETRI =====
print("⚙️ Parametri di generazione:")
print(f"DEVICE        : {DEVICE}")
print(f"AUDIO_PROMPT  : {AUDIO_PATH}")
print(f"EXAGGERATION  : {EXAGGERATION}")
print(f"CFG_WEIGHT    : {CFG_WEIGHT}")
print(f"LANGUAGE      : {LANGUAGE}")
print(f"SLOW          : {SLOW} (rate={SLOW_RATE})")
print(f"WHISPER       : {WHISPER}")
print(f"Paragrafi da generare: {total_paragraphs}\n")

# ===== INIZIALIZZA MODELLO TTS =====
model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)

all_wavs = []

for idx, para in enumerate(paragraphs):
    progress_pct = (idx / total_paragraphs) * 100
    print(f"🎤 Generating paragraph {idx+1}/{total_paragraphs} ({progress_pct:.1f}%)...")
    
    wav = model.generate(
        para,
        audio_prompt_path=AUDIO_PATH,
        exaggeration=EXAGGERATION,
        cfg_weight=CFG_WEIGHT,
        language_id=LANGUAGE
    )
    all_wavs.append(wav)
    
    # Salva il singolo WAV parziale
    ta.save(f"part-{idx+1}.wav", wav, model.sr)

# ===== CONCATENA TUTTI I WAV =====
full_wav = torch.cat(all_wavs, dim=1)
ta.save("output_full.wav", full_wav, model.sr)
print("✅ Tutti i paragrafi uniti in output_full.wav")

# ===== RALLENTAMENTO OPZIONALE =====
if SLOW:
    tfm = sox.Transformer()
    tfm.tempo(SLOW_RATE)
    tfm.build("output_full.wav", "output_full_slow.wav")
    print(f"✅ Tutti i paragrafi rallentati e uniti in output_full_slow.wav")

# ===== WHISPER OPZIONALE =====
if WHISPER:
    whisper_model = WhisperModel("small")
    segments, info = whisper_model.transcribe("output_full.wav", beam_size=5)
    print("📜 Trascrizione Whisper:")
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s]: {segment.text}")
