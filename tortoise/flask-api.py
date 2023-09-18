# flask-app.py
from flask import Flask, send_file, request, render_template
import json

import argparse
import os

import torch
import torchaudio

from api import TextToSpeech, MODELS_DIR
from utils.audio import load_voices

# create a Flask instance
app = Flask(__name__)

# a simple description of the API written in html.
# Flask can print and return raw text to the browser. 
# This enables html, json, etc. 
cuda_on = torch.cuda.is_available()
description =   f"""
                <!DOCTYPE html>
                <head>
                <title>API Landing</title>
                </head>
                <body>  
                    <h3>A simple API using Flask torch status: {cuda_on}</h3>
                    <a href="http://localhost:5000/tts?test=True&text=hi">Test request</a>
                    <br>
                    <a href="http://localhost:5000/tts?text=hi">Sample request</a>
                </body>
                """
				
# Routes refer to url'
# our root url '/' will show our html description
@app.route('/', methods=['GET'])
def hello_world():
    # return a html format string that is rendered in the browser
    return description

# Welcome back my

# our '/api' url
# requires user integer argument: value
# returns error message if wrong arguments are passed.
@app.route('/api', methods=['GET'])
def square():
    if not all(k in request.args for k in (["value"])):
        # we can also print dynamically 
        # using python f strings and with 
        # html elements such as line breaks (<br>)
        error_message =     f"\
                            Required paremeters : 'value'<br>\
                            Supplied paremeters : {[k for k in request.args]}\
                            "
        return error_message
    else:
        # assign and cast variable to int
        value = int(request.args['value'])
        # or use the built in get method and assign a type
        # http://werkzeug.palletsprojects.com/en/0.15.x/datastructures/#werkzeug.datastructures.MultiDict.get
        value = request.args.get('value', type=int)
        try:
            a = torch.cuda.is_available()
        except:
             a = "failed"
        return json.dumps({f"{a} | Value Squared" : value**2})
    


# Text to speech
def tts(
        text="Duck goes quack", 
        voice="random",
        preset="fast",
        use_deepspeed=False,
        kv_cache=True,
        half=True,
        output_path="../../../results/",
        model_dir=MODELS_DIR,
        candidates=1,
        seed=None,
        produce_debug_state=True,
        cvvp_amount=.0,
        test=False
        ):
    
    if test:
        print("Running test")
        # Create output directory if it does not exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Download audio and save it to output_path
        if not os.path.exists(os.path.join(output_path, 'gnome.wav')):
            import requests
            import shutil
            url = "https://github.com/bomxacalaka/testfiles/raw/main/gnome.wav"
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open('results/gnome.wav', 'wb') as f:
                    for chunk in r.iter_content(chunk_size=128):
                        f.write(chunk)
                print("File downloaded")
            else:
                print("Failed to download file")
    
    else:
        if torch.backends.mps.is_available():
            use_deepspeed = False
        os.makedirs(output_path, exist_ok=True)
        tts = TextToSpeech(models_dir=model_dir, use_deepspeed=use_deepspeed, kv_cache=kv_cache, half=half)

        selected_voices = voice.split(',')
        for k, selected_voice in enumerate(selected_voices):
            if '&' in selected_voice:
                voice_sel = selected_voice.split('&')
            else:
                voice_sel = [selected_voice]
            voice_samples, conditioning_latents = load_voices(voice_sel)

            gen, dbg_state = tts.tts_with_preset(text, k=candidates, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                    preset=preset, use_deterministic_seed=seed, return_deterministic_state=True, cvvp_amount=cvvp_amount)
            if isinstance(gen, list):
                for j, g in enumerate(gen):
                    torchaudio.save(os.path.join(output_path, f'{selected_voice}_{k}_{j}.wav'), g.squeeze(0).cpu(), 24000)
            else:
                torchaudio.save(os.path.join(output_path, f'{selected_voice}_{k}.wav'), gen.squeeze(0).cpu(), 24000)

            if produce_debug_state:
                os.makedirs('debug_states', exist_ok=True)
                torch.save(dbg_state, f'debug_states/do_tts_debug_{selected_voice}.pth')

        


# our '/tts' url
@app.route('/tts', methods=['GET'])
def tts_api():
    if not all(k in request.args for k in (["text"])):
        # we can also print dynamically 
        # using python f strings and with 
        # html elements such as line breaks (<br>)
        error_message =     f"\
                            Required paremeters : 'text'<br>\
                            Supplied paremeters : {[k for k in request.args]}\
                            "
        return error_message
    else:
        # assign and cast variable to int
        text = request.args.get("text")
        voice = request.args.get('voice', "random")
        preset = request.args.get('preset', "fast")
        use_deepspeed = request.args.get('use_deepspeed', False)
        kv_cache = request.args.get('kv_cache', True)
        half = request.args.get('half', True)
        output_path = request.args.get('output_path', "results/")
        model_dir = request.args.get('model_dir', MODELS_DIR)
        candidates = request.args.get('candidates', 1)
        seed = request.args.get('seed', None)
        produce_debug_state = request.args.get('produce_debug_state', True)
        cvvp_amount = request.args.get('cvvp_amount', .0)
        test = request.args.get('test', False)
        
        tts(text=text, voice=voice, preset=preset, use_deepspeed=use_deepspeed, kv_cache=kv_cache, half=half, output_path=output_path, model_dir=model_dir, candidates=candidates, seed=seed, produce_debug_state=produce_debug_state, cvvp_amount=cvvp_amount, test=test)

        # Get all files in the output directory
        files = os.listdir(output_path)
        print(text, test, files)
        # Select the most recent file
        # filename = max(files, key=os.path.getctime)
        # Get the full path of the file
        file_path = os.path.join(output_path, files[0])

        # Return audio file to browser
        return send_file(
            file_path,
            as_attachment=True,  # This forces the browser to download the file.
            download_name=file_path,  # Specify the name of the downloaded file.
            mimetype='audio/wav'
        )






if __name__ == "__main__":
	# for debugging locally
	# app.run(debug=True, host='0.0.0.0',port=5000)
	
	# for production
	app.run(host='0.0.0.0', port=5000)