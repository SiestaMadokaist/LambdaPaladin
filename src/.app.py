from flask import Flask, Response, send_file
from PIL import Image
import os
import json
import sys
import CornerDetection as cd
app = Flask(__name__)

@app.route('/<x0>/<y0>/<x1>/<y1>')
def eigen(x0, y0, x1, y1):
    path = '../dataset/jaffe/KA.AN1.39.jpg'
    image = Image.open(path).crop(map(int, (x0, y0, x1, y1)))
    cd.detect(image, 100, True)
    return 'success'

@app.route('/debug')
def debug():
    return send_file('./debug.html')

@app.route('/face')
def face():
    return send_file('./face.jpg')

@app.route('/chess')
def chess():
    return send_file('./chess.bmp')

@app.route("/hello")
def hello():
    return send_file("./hello.jpg")

if __name__ == '__main__':    
    if len(sys.argv) < 2:
        pass
    else:
        app.run(debug=True)