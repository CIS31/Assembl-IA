import sys
import requests
from bs4 import BeautifulSoup
import re
import os
from datetime import datetime
from pathlib import Path

# Classe AzureUtils déjà définie dans ton script
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

# URL de base
base_url = "https://www.assemblee-nationale.fr"
url = base_url + "/dyn/17/comptes-rendus/seance"

# Récupérer la page principale
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# Fonction pour convertir une date française en objet datetime
def french_date_to_datetime(date_str):
    mois_fr = {
        'janvier': 1, 'février': 2, 'fevrier': 2, 'mars': 3, 'avril': 4,
        'mai': 5, 'juin': 6, 'juillet': 7, 'août': 8, 'aout': 8,
        'septembre': 9, 'octobre': 10, 'novembre': 11, 'décembre': 12, 'decembre': 12
    }
    parts = date_str.split('-')
    if len(parts) == 4:
        jour = int(parts[1])
        mois = mois_fr.get(parts[2].lower(), 0)
        annee = int(parts[3])
        if mois != 0:
            return datetime(annee, mois, jour)
    return None

# Extraire les liens et les dates
links_dates = []
for a in soup.find_all("a", href=True):
    href = a['href']
    if href.startswith("/dyn/17/comptes-rendus/seance/"):
        m = re.search(r"/([^/]+)$", href)
        if m:
            last = m.group(1)
            dm = re.search(r"du-([^-]+-\d{1,2}-[a-zéû]+-\d{4})", last)
            if dm:
                dstr = dm.group(1)
                dobj = french_date_to_datetime(dstr)
                if dobj:
                    links_dates.append((href, dstr, dobj))

if not links_dates:
    exit(1)

# Trier les liens par date décroissante
links_dates.sort(key=lambda x: x[2], reverse=True)

# Obtenir les dates uniques
unique_dates = []
seen = set()
for _, dstr, _ in links_dates:
    if dstr not in seen:
        unique_dates.append(dstr)
        seen.add(dstr)

if len(unique_dates) < 2:
    exit(1)

# Obtenir le lien correspondant à la deuxième date
second_date_str = unique_dates[1]
matching_links = set()
for href, dstr, _ in links_dates:
    if dstr == second_date_str:
        h = href.split("#")[0]
        if h.endswith(".pdf"):
            h = h[:-4]
        matching_links.add(h)

if not matching_links:
    exit(1)

# Récupérer le dernier lien correspondant
last_link = sorted(matching_links)[-1]
response2 = requests.get(base_url + last_link)
soup2 = BeautifulSoup(response2.content, "html.parser")

# Trouver le lien XML
xml_link = None
for a in soup2.find_all("a", href=True):
    href = a['href']
    if href.endswith(".xml") and "opendata" in href:
        xml_link = href
        break

if not xml_link:
    exit(1)

# Télécharger le fichier XML
full_xml_url = xml_link if xml_link.startswith("http") else base_url + xml_link
xml_response = requests.get(full_xml_url)
if xml_response.status_code == 200:
    filename = os.path.basename(full_xml_url)

    # Initialiser AzureUtils
    azure_utils = AzureUtils(mount_dir="/mnt/data")
    AZURE_RUN = azure_utils.detect_azure_run()

    if AZURE_RUN:
        # Si on est dans Azure, monter le répertoire et copier le fichier
        azure_utils.mount_dir_Azure()
        tmp_path = Path("/tmp") / filename
        with open(tmp_path, "wb") as f:
            f.write(xml_response.content)
        dest = f"{azure_utils.mount_dir}/text/input/WebScrapping_{filename}"
        dbutils.fs.mkdirs(f"{azure_utils.mount_dir}/text/input")
        
        # Copier le fichier sans l'argument overwrite
        dbutils.fs.cp(f"file:{tmp_path}", dest)

        print(f"[Blob] Fichier XML sauvegardé dans Azure → {dest}")
    else:
        # Si on est en local, sauvegarder le fichier localement
        input_dir = Path("/dbfs/mnt/data/text/input")
        input_dir.mkdir(parents=True, exist_ok=True)
        file_path = input_dir / f"WebScrapping_{filename}"
        with open(file_path, "wb") as f:
            f.write(xml_response.content)
        print(f"[Local] Fichier XML téléchargé → {file_path}")
else:
    print(f"Erreur {xml_response.status_code}")