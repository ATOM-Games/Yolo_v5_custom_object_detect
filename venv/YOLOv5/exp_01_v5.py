import base64
import os
import sys
from pathlib import Path
import cv2
import numpy
import torch
import torch.backends.cudnn as cudnn
from PIL import Image as im
import io
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

device = select_device('сpu')
model = DetectMultiBackend('Models/Best_weapon.pt', device=device, dnn=False, data='Models/ABC_weapon_01.yaml')
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size((640, 640), s=model.stride)  # check image size
half = False
half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
if pt or jit:
    model.model.half() if half else model.model.float()
elif engine and model.trt_fp16_input != half:
    LOGGER.info('model ' + (
        'requires' if model.trt_fp16_input else 'incompatible with') + ' --half. Adjusting automatically.')
    half = model.trt_fp16_input



detStatus = 'OFF'
res_image_ND = None
res_image_YD = None

def getDetStatus() :
    global detStatus
    return detStatus
def setDetStatus(neDetStatus) :
    global detStatus
    detStatus = neDetStatus
def getResImg():
        global res_image_ND
        global res_image_YD
        return res_image_ND, res_image_YD
def setResImg(new_img_ND, new_img_YD):
    global res_image_ND
    global res_image_YD
    res_image_ND = ndr_to_bs(new_img_ND)
    res_image_YD = ndr_to_bs(new_img_YD)

def ndr_to_bs(ndr):
    if ndr is None : return None
    ndrd = io.BytesIO()
    im.fromarray(ndr).convert('RGB').save(ndrd, format='JPEG')
    return base64.b64encode(ndrd.getvalue()).__str__()

def ndr_to_bs2(ndr):
    return base64.b64encode(im.fromarray(ndr).convert('RGB').tobytes()).__str__()

#model.warmup(imgsz=imgsz, half=half)

def Detector(camera, s1, s2):
    if getDetStatus() == 'OFF': setDetStatus('Wait')
    #cudnn.benchmark = True
    dts = LoadStreams(camera, img_size=imgsz, stride=stride, auto=pt)
    for path, im, im0s, vid_cap, s in dts: # аналог while True y OpenCV
        if getDetStatus() == 'OFF':
            setResImg(None, None)
            break
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, s1, s2, None, False, max_det=1000)
        for i, det in enumerate(pred):
            p, im0, frame = path[i], im0s[i].copy(), dts.count
            print(getDetStatus())
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c]
                    annotator.box_label(xyxy, label, color=colors(c, True))
            im0 = annotator.result()
            if getDetStatus() == 'Wait' : setDetStatus('Detect')
            if getDetStatus() == 'OFF':
                setResImg(None, None)
                break
            if getDetStatus() == 'Detect' and im0s[i] is not None and im0 is not None : setResImg(im0s[i], im0)
    print("SSSTOOOPPEEED")