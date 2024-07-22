from fastapi import FastAPI ,UploadFile, File
from image_processing_core import cropPerspective ,get_answer
from PIL import Image
import uvicorn
from typing import List
import uuid
import json

class Answer:
    answer: List

app = FastAPI(debug=True)
IMAGEDIR = "images/"
IMAGECROPDIR = "crop/"

@app.get('/')
async def hello_world():
    return 'services is online.'

@app.get('/response')
async def response():
    with open('assets/response.json', encoding="utf8") as f:
        d = json.load(f)
        return d

@app.post('/predict')
async def predict_image(file:UploadFile = File(...)):
    print("tasedasdasd ", file)
    # Get image and save file to images folder
    file.filename = f"{uuid.uuid4()}.jpg"
    contents= await file.read()
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)
        
    numpy_array = cropPerspective(f"{IMAGEDIR}{file.filename}")
    im = Image.fromarray(numpy_array)
    im_crop = f"{IMAGECROPDIR}{uuid.uuid4()}.jpg"
    im.save(im_crop)
    id , answer = get_answer(im_crop)
    
    return {"id":id,"answer":answer,"headers":file.headers ,"filename": file.filename}


if __name__ == "__main__":
    uvicorn.run(app,host='0.0.0.0',port=4000)