import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json


def return_boxes(net, image, iou_thresh, nms_thresh, LABELS, threshold, state):

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    image = cv2.resize(image, (640, 480))   # dtype=int, [0...255]
    image1 = image.astype(np.float32)/255.0 # [0...1]

    #Depth
    rate = 0.3  # rate of the rectangle in the middle of each bounding box to estimate better depth
    res = (1-rate)/2
    # load json and create model
    json_file = open('./../models_depth/Unet/2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("./../models_depth/Unet/2.h5")
    print("Loaded model from disk")
    dep_image = model.predict(np.expand_dims(image1, axis=0))[0].squeeze()
    dep_resized = cv2.resize(dep_image, (640, 480)).astype(np.float32)

    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # show timing information on YOLO
    detection_time = end - start


    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > iou_thresh:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, iou_thresh, nms_thresh)

    if len(idxs) > 0:
        for i in idxs.flatten():

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # Depth
            depth_list = []
            for ii in range(int(x+res*w), int(x+(res+rate)*w)):
              for j in range(int(y+res*h), int(y+(res+rate)*h)):
                depth_list.append(dep_resized[j, ii])
            Depth = (sum(depth_list) / len(depth_list)) * 9.99547

            if(state == 'MORE'):
              if(Depth >= threshold):
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                #text = "{}: {:.2f}, Depth : {:.2f}".format(LABELS[classIDs[i]], confidences[i], Depth)
                text = "{}: {:.2f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                text = "Depth : {:.2f}".format(Depth)
                cv2.putText(image, text, (x, y + h + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if(state == 'LESS'):
              if(Depth < threshold):
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                #text = "{}: {:.2f}, Depth : {:.2f}".format(LABELS[classIDs[i]], confidences[i], Depth)
                text = "{}: {:.2f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                text = "Depth : {:.2f}".format(Depth)
                cv2.putText(image, text, (x, y + h + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #cv2.imshow("Image", image)

    _, t = plt.subplots(1, 1)
    t.imshow(image)
    plt.axis('off')
    plt.savefig("Object_Detection/static/test.jpeg", dpi=360, bbox_inches='tight')

    return detection_time
