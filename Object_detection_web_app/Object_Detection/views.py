from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt


import io
from PIL import Image
import cv2
import numpy as np
from base64 import b64decode
from .utils import *


@csrf_exempt
def object_detection_api(api_request):
    json_object = {'success': False}
    
    threshold = 4.0
    state = 'MORE'

    if api_request.method == "POST":

        if api_request.POST.get("image64", None) is not None:
            base64_data = api_request.POST.get("image64", None).split(',', 1)[1]
            data = b64decode(base64_data)
            data = np.array(Image.open(io.BytesIO(data)))
            detection_time = detection(data, threshold, state)

        elif api_request.FILES.get("image", None) is not None:
            image_api_request = api_request.FILES["image"]
            image_bytes = image_api_request.read()
            image = Image.open(io.BytesIO(image_bytes))
            detection_time = detection(image, threshold, state)

    json_object['success'] = True
    json_object['time'] = str(round(detection_time))+" seconds"
    print(json_object)
    return JsonResponse(json_object)


def detect_request(api_request):
    return render(api_request, 'index.html')


def detection(original_image, threshold=4.0, state='MORE'):
    cfg_file = './yolov3.cfg'
    weight_file = './yolov3.weights'
    names = './coco.names'

    m = cv2.dnn.readNetFromDarknet(cfg_file, weight_file)
    class_names = open(names).read().strip().split("\n")

    nms_thresh = 0.6
    iou_thresh = 0.4

    detection_time = return_boxes(m, original_image, iou_thresh, nms_thresh, class_names, threshold, state)

    return detection_time
