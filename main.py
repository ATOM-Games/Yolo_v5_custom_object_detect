import PIL.Image
import cv2
from YOLOv5.exp_01_v5 import Detector, setDetStatus, getDetStatus, getResImg, setResImg, DetMy
from flask import Flask, jsonify, request, make_response, render_template,json
from flask_cors import CORS
import requests
from PIL import Image
import numpy as np
from base64 import  b64decode
from cv2 import imdecode, IMREAD_COLOR
import sys



params = dict()
for line in open("ServerConfig.cfg", "r").readlines():
    p_name, p_val = line.split("=")
    params[p_name.strip()] = p_val.strip()
app = Flask(__name__, template_folder='web', static_folder='web')
CORS(app)

global_result = None

camera = '10.0.0.195:10001'

listeners = []

last_res = None;

def NewDetect(s1, s2):
    setDetStatus('Wait')
    global last_res
    while True:
        b64_frame = requests.post('http://'+camera+'/get_b64_frame').content
        cv_frame = b64_to_cv(b64_frame)
        cv_frame = cv2.resize(cv_frame, (640,640))
        img = DetMy(cv_frame, s1, s2)
        if img is not []:
            #запросик
            last_res = img
            json_message = []
            json_message.append({
                "name": "weapon",
                "res": img
            })
            print(json_message)
            for listener in listeners:
            requests.post(listener, json=json_message).content
        else:
            if last_res is not None:
                last_res = None
                json_message = []
                json_message.append({
                    "name": "weapon",
                    "res": []
                })
                print(json_message)
                for listener in listeners:
                requests.post(listener, json=json_message).content
            print("zapros")
        if getDetStatus() == 'Wait': setDetStatus('Detect')
        if getDetStatus() == 'OFF':
            #setResImg(None, None)
            break
    print("SSSTOOOPPEEED")

def b64_to_cv(b64):
    _bytes = b64decode(b64)
    np_arr = np.frombuffer(_bytes, dtype=np.uint8)
    cv = imdecode(np_arr, flags=IMREAD_COLOR)
    return cv




@app.route('/')
def index():
    return """<h1>I am Alive</h1>"""

@app.route('/admin', methods=['GET'])
def admin():
    return render_template("templates/admin/index.html")

@app.route('/status', methods=['POST'])
def status():
    det = request.get_json(force=True)['detect']
    #can = request.get_json(force=True)['camera'].__str__()
    s1 = float(request.get_json(force=True)['step_01'])
    s2 = float(request.get_json(force=True)['step_02'])
    if(getDetStatus()=='OFF') and det=='start':
        #Detector(can, s1, s2)
        NewDetect(s1, s2)
    if(getDetStatus()=='Detect') and det=='stop':
        setDetStatus('OFF')
    return """<h1>I am Alive</h1>"""

@app.route('/getStatus', methods=['POST'])
def getStatus():
    return jsonify({ "status": getDetStatus() })



@app.route('/resultDetect', methods=['POST'])
def resultDetect():
    det = request.get_json(force=True).get("address")
    if det:
        res = requests.post(det).json()
        return jsonify({"res_img": res["res_img"], "status": "Detect", "detectes": []})
    else:
        ND, dets = getResImg()
        return jsonify({ "res_img": ND, "status" : getDetStatus(), "detectes": dets })

@app.route('/www', methods=['POST'])
def wwwww():
    print(request.json)
    print(request.data)
    print(request.form)
    return 200

@app.route('/subscribe', methods=['POST'])
def subscribe():
    global listeners
    det = request.get_json(force=True).get("address")
    det2 = request.get_json(force=True).get("method")
    print("!!!!!"+det+" "+det2)
    if(det in listeners):
        print('have')
    else :
        listeners.append(det)
    print(listeners)
    return jsonify({ 'm' : 'm' })




if __name__ == '__main__':
    app.run(host=params["address"], port=params["port"], debug=bool(params["debug"]))