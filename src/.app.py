from flask import Flask, Response, send_file
from PIL import Image
import os
import json
import sys
import CornerDetection as cd
app = Flask(__name__)

@app.route('/<x0>/<y0>/<x1>/<y1>')
def eigen(x0, y0, x1, y1):
    path = 'img/input.jpg'    
    image = Image.open(path).crop(map(int, (x0, y0, x1, y1)))
    image.save("./img/middle.jpg")
    eigen = cd.detect(image, 100, True)    
    return 'success'

@app.route('/')
def debug():
    return send_file('./index.html')

@app.route('/input')
def inputImage():
    return send_file('./img/input.jpg')

@app.route("/middle")
def middleImage():
    return send_file('./img/middle.jpg')

@app.route('/output')
def outputImage():
    return send_file('./img/output.jpg')

if __name__ == '__main__':    
    if len(sys.argv) < 2:
        pass
    else:
        app.run(debug=True)