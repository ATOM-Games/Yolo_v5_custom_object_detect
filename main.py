from YOLOv5.exp_01_v5 import Detector, setDetStatus, getDetStatus, getResImg, setResImg
from flask import Flask, jsonify, request, make_response, render_template
from flask_cors import CORS

params = dict()
for line in open("ServerConfig.cfg", "r").readlines():
    p_name, p_val = line.split("=")
    params[p_name.strip()] = p_val.strip()
app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return """<h1>I am Alive</h1>"""

@app.route('/status', methods=['POST'])
def status():
    det = request.get_json(force=True)['detect']
    can = request.get_json(force=True)['camera'].__str__()
    s1 = float(request.get_json(force=True)['step_01'])
    s2 = float(request.get_json(force=True)['step_02'])
    if(getDetStatus()=='OFF') and det=='start':
        Detector(can, s1, s2)
    if(getDetStatus()=='Detect') and det=='stop':
        setDetStatus('OFF')

    return """<h1>I am Alive</h1>"""

@app.route('/getStatus', methods=['POST'])
def getStatus():
    return jsonify({ "status": getDetStatus() })



@app.route('/resultDetect', methods=['POST'])
def resultDetect():
    det = request.get_json(force=True)['det']
    ND, YD = getResImg()
    res = 'None'
    if det == 'detect' :
        res = YD
    else :
        res = ND
    return jsonify({ "res_img": res, "status" : getDetStatus() })






if __name__ == '__main__':
    app.run(host=params["address"], port=params["port"], debug=bool(params["debug"]))