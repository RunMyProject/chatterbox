# ===================
# = slow_down_wav.py 🎧 =
# ===================
import torchaudio
import torchaudio.transforms as T

INPUT_FILE = "output_full.wav"
OUTPUT_FILE = "output_full_smooth.wav"
SLOW_FACTOR = 0.65  # 65% della velocità originale

# Carica audio
wav, sr = torchaudio.load(INPUT_FILE)

# Applica rallentamento con pitch invariato
tempo_transform = T.TimeStretch()
# Torchaudio richiede complesso STFT, quindi useremo Sox-style
# Alternativa semplice con SoX integrato:
import subprocess

subprocess.run([
    "sox", INPUT_FILE, OUTPUT_FILE,
    "tempo", "-s", str(SLOW_FACTOR)
])

print(f"✅ Audio rallentato salvato in {OUTPUT_FILE}")
