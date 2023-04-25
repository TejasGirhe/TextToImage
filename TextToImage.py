from colabcode import ColabCode
cc = ColabCode(port = 12000, code = False)
from fastapi import FastAPI
import os
from fastapi.responses import FileResponse
from PIL import ImageTk
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

import pyrebase
firebaseConfig = {
  "apiKey": "AIzaSyD4OKbsEvaBQF5QCo-A9TikXE9-8XIquPE",
  "authDomain": "text-to-image-c4dd6.firebaseapp.com",
  "projectId": "text-to-image-c4dd6",
  "storageBucket": "text-to-image-c4dd6.appspot.com",
  "messagingSenderId": "1060244681808",
  "appId": "1:1060244681808:web:487a0706a945ca90f84d55",
  "measurementId": "G-355QF81HNT",
  "serviceAccount" : "/content/sample_data/text-to-image.json",
  "databaseURL" : "https://text-to-image-c4dd6-default-rtdb.firebaseio.com/"
}

firebase = pyrebase.initialize_app(firebaseConfig)


auth_token = "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"
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
path = "/content/sample_data"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token) 
pipe.to(device) 

app = FastAPI()

@app.get("/")
async def generate(prompt_text: str): 
    with autocast(device): 
        image = pipe(prompt_text, guidance_scale=8.5)
    
    # image.save('/content/sample_data/generatedimage.png')
    # image_path = open(image, "rb")
    img = image.images[0]
    img.save('/content/sample_data/image.png', 'PNG')
    firebase.storage().child(prompt_text + ".png").put('/content/sample_data/image.png')
    return {"Id" : prompt_text}
    # return {"image": "Success"}