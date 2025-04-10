import os
import argparse
from dotenv import load_dotenv
import torch
from pyannote.audio import Pipeline
import matplotlib.pyplot as plt
from pyannote.core import Segment
from pyannote.audio.pipelines import ProgressHook

load_dotenv()

hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Vérifier si CUDA (GPU) est disponible
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
print(f"Utilisation de {device} pour la diarisation.")

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", 
    use_auth_token=hf_token)

def parse_args():
    parser = argparse.ArgumentParser(description="Diarisation de locuteurs dans un fichier audio.")
    parser.add_argument("audio_file", type=str, help="Chemin vers le fichier audio (.wav)")
    return parser.parse_args()


args = parse_args()

audio_file = args.audio_file

# Get a progress bar
with ProgressHook() as hook:
    diarization = pipeline(audio_file, hook=hook)

output_text_file = "diarization_results.txt"


speakers_intervals = {}
for speech_turn in diarization.itertracks(yield_label=True):
    speaker, segment = speech_turn
    if speaker not in speakers_intervals:
        speakers_intervals[speaker] = []
    speakers_intervals[speaker].append(segment)

with open(output_text_file, "w") as f:
    for speaker, intervals in speakers_intervals.items():
        f.write(f"Locuteur {speaker}:\n")
        for interval in intervals:
            f.write(f"  Début = {interval.start:.2f} s, Fin = {interval.end:.2f} s\n")
        f.write("\n")

# Nombre de locuteurs détectés
num_speakers = len(speakers_intervals)
print(f"Nombre de locuteurs détectés : {num_speakers}")

fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.get_cmap("tab10", num_speakers) 


for idx, (speaker, intervals) in enumerate(speakers_intervals.items()):
    for interval in intervals:
        ax.plot([interval.start, interval.end], [idx, idx], color=colors(idx), lw=6, label=f"Locuteur {speaker}" if interval == intervals[0] else "")

ax.set_yticks(range(num_speakers))
ax.set_yticklabels([f"Locuteur {speaker}" for speaker in speakers_intervals.keys()])
ax.set_xlabel("Temps (s)")
ax.set_title("Diarisation des locuteurs (Intervalles de parole)")
ax.legend()
plt.tight_layout()
plt.savefig("../assets/diarization_plot.png")  
plt.close()

print(f"Résultats enregistrés dans '{output_text_file}' et graphique sauvegardé sous 'diarization_plot.png'")
