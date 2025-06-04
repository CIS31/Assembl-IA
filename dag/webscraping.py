import requests
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import re
import subprocess

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
    # video_url = "https://videos.assemblee-nationale.fr/video.16927164_6835a10eb1ebe.commission-du-developpement-durable--programmation-nationale-et-simplification-normative-dans-le-se-27-mai-2025"
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

# html_script = """<script src="https://cdn.jsdelivr.net/hls.js/latest/hls.min.js"></script><video id="hls-player" style="width:640px;height:360px;" controls=""></video><script>var player = document.getElementById('hls-player'); var url = 'https://anorigin.vodalys.com/videos/definst/mp4/ida/domain1/2025/05/6238_20250527132503_1.mp4/master.m3u8'; if(Hls.isSupported()) {var hls = new Hls(); hls.loadSource(url); hls.attachMedia(player); } else {player.setAttribute('src', url); } player.addEventListener('loadeddata', function() {player.currentTime = 0;} );</script>"""

# Extract the URL from var url = '...';
match = re.search(r"var\s+url\s*=\s*'([^']+)'", embed_code)

if match:
    m3u8_url = match.group(1)
    print(" Extracted URL:", m3u8_url)
else:
    print(" URL not found")

# m3u8_url = "https://anorigin.vodalys.com/videos/definst/mp4/ida/domain1/2025/05/6238_20250527132503_1.mp4/master.m3u8"
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

