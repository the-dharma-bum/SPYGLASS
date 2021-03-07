import os
import subprocess

DATASET_URL = 'https://mega.nz/#!EAE00ZqC!eRPUTgpVMR3z7UKbLV0_BXqbvkSYF16yWIfCb9kcZfQ'


def setup():
  print('Downloading github repository...')
  os.system('git clone https://github.com/the-dharma-bum/SPYGLASS/')
  os.chdir('SPYGLASS')
  os.system('git checkout -b fusion')
  os.system('git branch --set-upstream-to=origin/fusion fusion')
  os.system('git pull -q')
  print('Downloading requirements...')
  os.system('pip install -q -r requirements.txt')


def get_data():
    print("Downloading dataset ...")
    os.system('apt install jq pv')
    os.system('chmod 755 /content/SPYGLASS/colab/download.sh')
    subprocess.check_call(['/content/SPYGLASS/colab/download.sh', DATASET_URL])
    print("Extracting dataset ...")
    os.system('unzip -q SPYGLASS_cropped.zip')