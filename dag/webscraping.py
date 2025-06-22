import requests
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import re
import subprocess
from pathlib import Path
import os

# Charger la page d'accueil des vidéos
url_base = "https://videos.assemblee-nationale.fr"
response = requests.get(url_base)
soup = BeautifulSoup(response.content, "html.parser")

# Trouver les vidéos dans le premier div.row sous .carousel-inner > .item.active
carousel = soup.select_one(".carousel-inner .item.active #row0")
videos = carousel.find_all("div", class_="span4")

# Vérifie qu'on a au moins deux vidéos
if len(videos) < 2:
    raise Exception("Moins de deux vidéos trouvées.")

# Extraire le lien de l’avant-dernière vidéo
avant_dernier_video = videos[0]  # [0] = dernière, [1] = avant-dernière

link_tag = avant_dernier_video.find("a", class_="vl")
print(" Lien de la vidéo :", link_tag["href"])
video_relative_url = link_tag["href"]
video_url = f"{url_base}/{video_relative_url}"
print(" URL complète de la vidéo :", video_url)

driver = webdriver.Chrome()

time.sleep(2)
try:
    # Ouvrir la page de la vidéo
    driver.get(video_url)

    # Attendre que la page charge le contenu dynamiquement (pause simple ici, tu peux aussi utiliser WebDriverWait)
    time.sleep(5)

    # Trouver le textarea contenant le code embed
    textarea = driver.find_element(By.CSS_SELECTOR, "div.link textarea")
    embed_code = textarea.get_attribute("value") or textarea.text

    print(" Code embed récupéré :\n")
    print(embed_code)

finally:
    driver.quit()

# Extract the URL from var url = '...';
match = re.search(r"var\s+url\s*=\s*'([^']+)'", embed_code)

if match:
    m3u8_url = match.group(1)
    print(" Extracted URL:", m3u8_url)
else:
    print(" URL not found")
    exit(1)

output_file = "output_segment.mp4"
start_time = "278.38"
duration = "60"  # 1 minute

command = [
    "ffmpeg",
    "-ss", start_time,        # Start at 278.38 seconds
    "-i", m3u8_url,
    "-t", duration,           # Record only 60 seconds
    "-c", "copy",             # Copy without re-encoding
    output_file
]

try:
    subprocess.run(command, check=True)
    print(" Téléchargement terminé :", output_file)
except subprocess.CalledProcessError as e:
    print(" Erreur lors du téléchargement :", e)
    exit(1)

# Copier la vidéo dans Azure ou localement
try:
    from emotion_detectionvf import AzureUtils
except ImportError:
    AzureUtils = None

if AzureUtils is not None:
    azure_utils = AzureUtils(mount_dir="/mnt/data")
    AZURE_RUN = azure_utils.detect_azure_run()
else:
    AZURE_RUN = False

if AZURE_RUN:
    azure_utils.mount_dir_Azure()
    tmp_path = Path("/tmp") / output_file
    dest = f"{azure_utils.mount_dir}/video/input/WebScrapping_{output_file}"
    dbutils.fs.mkdirs(f"{azure_utils.mount_dir}/video/input")
    dbutils.fs.cp(f"file:{tmp_path}", dest, overwrite=True)
    print(f"[Blob] Vidéo sauvegardée dans Azure → {dest}")
else:
    input_dir = Path("/dbfs/mnt/data/video/input")
    input_dir.mkdir(parents=True, exist_ok=True)
    file_path = input_dir / f"WebScrapping_{output_file}"
    with open(file_path, "wb") as f:
        with open(output_file, "rb") as src:
            f.write(src.read())
    print(f"[Local] Vidéo téléchargée → {file_path}")