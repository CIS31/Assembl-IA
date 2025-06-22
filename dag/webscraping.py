import sys
import requests
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import re
import subprocess
from pathlib import Path

class AzureUtils:
    def __init__(self, mount_dir):
        self.mount_dir = mount_dir

    def detect_azure_run(self):
        """
        Function to detect if the code is running in an Azure environment.
        """
        args = dict(arg.split('=') for arg in sys.argv[1:] if '=' in arg)
        return args.get("AZURE_RUN", "false").lower() == "true"

    def mount_dir_Azure(self):
        """
        Function to mount the directory in Azure environment.
        """
        def is_mounted(mount_point):
            mounts = [mount.mountPoint for mount in dbutils.fs.mounts()]
            return mount_point in mounts

        configs = {
            "fs.azure.account.auth.type": "OAuth",
            "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
            "fs.azure.account.oauth2.client.id": dbutils.secrets.get(scope="az-kv-assemblia-scope", key="sp-application-id"),
            "fs.azure.account.oauth2.client.secret": dbutils.secrets.get(scope="az-kv-assemblia-scope", key="sp-secret-value"),
            "fs.azure.account.oauth2.client.endpoint": f"https://login.microsoftonline.com/{dbutils.secrets.get(scope='az-kv-assemblia-scope', key='sp-tenant-id')}/oauth2/token"
        }

        if not is_mounted(self.mount_dir):
            dbutils.fs.mount(
                source="abfss://data@azbstelecomparis.dfs.core.windows.net/",
                mount_point=self.mount_dir,
                extra_configs=configs
            )
            print(f"Successfully mounted {self.mount_dir}")
        else:
            print(f"{self.mount_dir} is already mounted")

# Configurer les options de Chrome
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")  # Nécessaire pour Databricks
chrome_options.add_argument("--disable-dev-shm-usage")  # Réduit les problèmes de mémoire
chrome_options.add_argument("--headless")  # Mode sans interface graphique

# Utilisation dynamique du bon chromedriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

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

output_file = "/tmp/output_segment.mp4"

command = [
    "ffmpeg",
    "-y",                     # Force écrasement du fichier de sortie
    "-i", m3u8_url,
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
azure_utils = AzureUtils(mount_dir="/mnt/data")
AZURE_RUN = azure_utils.detect_azure_run()

if AZURE_RUN:
    azure_utils.mount_dir_Azure()
    tmp_path = Path("/tmp") / output_file
    dest = f"{azure_utils.mount_dir}/video/input/videos/WebScrapping_output_segment.mp4"
    dbutils.fs.mkdirs(f"{azure_utils.mount_dir}/video/input/videos")
    
    # Supprimer le fichier de destination s'il existe
    try:
        dbutils.fs.rm(dest)
    except Exception as e:
        print(f"Le fichier {dest} n'existe pas ou ne peut pas être supprimé : {e}")

    # Copier le fichier
    dbutils.fs.cp(f"file:{tmp_path}", dest)
    print(f"[Blob] Vidéo sauvegardée dans Azure → {dest}")
else:
    input_dir = Path("/dbfs/mnt/data/video/input")
    input_dir.mkdir(parents=True, exist_ok=True)
    file_path = input_dir / f"WebScrapping_{output_file}"
    with open(file_path, "wb") as f:
        with open(output_file, "rb") as src:
            f.write(src.read())
    print(f"[Local] Vidéo téléchargée → {file_path}")