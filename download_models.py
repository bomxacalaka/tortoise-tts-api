import os
import urllib.request
import argparse
import sys
import time

MODELS = {
    'autoregressive.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/autoregressive.pth',
    'classifier.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/classifier.pth',
    'clvp2.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/clvp2.pth',
    'cvvp.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/cvvp.pth',
    'diffusion_decoder.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/diffusion_decoder.pth',
    'vocoder.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/vocoder.pth',
    'rlg_auto.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_auto.pth',
    'rlg_diffuser.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_diffuser.pth',
}

def download_file(url, file_path, chunk_size=1024):
    response = urllib.request.urlopen(url)
    file_size = int(response.getheader('content-length', 0))
    downloaded_size = 0
    start_time = time.time()
    with open(file_path, 'wb') as file:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            file.write(chunk)
            downloaded_size += len(chunk)
            yield downloaded_size, file_size, time.time() - start_time

def format_size(size):
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
    index = 0
    while size > 1024 and index < len(suffixes) - 1:
        size /= 1024.0
        index += 1
    return f"{size:.2f} {suffixes[index]}"

def download_models(models_dir):
    os.makedirs(models_dir, exist_ok=True)
    for model_name, model_url in MODELS.items():
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            print(f'Downloading {model_name}...')
            progress_bar_length = 50
            for downloaded_size, file_size, elapsed_time in download_file(model_url, model_path):
                percent_complete = downloaded_size / file_size
                arrow = '=' * int(round(progress_bar_length * percent_complete))
                spaces = ' ' * (progress_bar_length - len(arrow))
                download_speed = downloaded_size / elapsed_time if elapsed_time > 0 else 0
                sys.stdout.write(f'\r[{arrow}{spaces}] {int(percent_complete * 100)}% '
                                 f'({format_size(downloaded_size)}/{format_size(file_size)}) '
                                 f'Speed: {format_size(download_speed)}/s')
                sys.stdout.flush()
            print("\nDownload completed!")
        else:
            print(f'{model_name} already exists, skipping download.')

if __name__ == '__main__':
    download_models("models")
