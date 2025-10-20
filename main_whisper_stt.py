# =========================
# = main_whisper_stt.py 📜 =
# =========================

from faster_whisper import WhisperModel

# ===== CONFIG =====
AUDIO_FILE = "ElevenLabs_2025-10-18T22_33_07_Yuki - ASMR Whisper_pvc_sp106_s50_sb70_se50_b_m2.mp3"
MODEL_SIZE = "small"
OUTPUT_FILE = "output.txt"

# ===== INIZIALIZZA WHISPER =====
whisper_model = WhisperModel(MODEL_SIZE)

# ===== TRASCRIZIONE =====
segments, info = whisper_model.transcribe(AUDIO_FILE, beam_size=5)

# ===== SCRIVE SU FILE =====
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("📜 Trascrizione Whisper (STT, solo controllo):\n")
    for segment in segments:
        f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s]: {segment.text}\n")

print(f"✅ Trascrizione salvata in {OUTPUT_FILE}")
