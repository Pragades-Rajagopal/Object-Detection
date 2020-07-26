from numpy.core.fromnumeric import size
from cv2 import cv2
import numpy as np

#loading yolov3-tiny
net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')

# Initializing webcam
cap = cv2.VideoCapture(0) #webcam is in channel-0

classes = []
with open('coco.names', 'r') as file:
    classes = [(lines.strip()) for lines in file.readlines()]

layer_names = net.getLayerNames()
output_layers = [(layer_names[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
while True:
    _, frame = cap.read()

    width, height, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    boxes = []
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    num_of_Obj_detected = len(boxes)
    for i in range(num_of_Obj_detected):
        if i in indexes:
            x, y, w, h = boxes[i]
            labels = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
            cv2.putText(frame, labels + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)


    cv2.imshow('Image', frame)
    key = cv2.waitKey(1)
    if key == 10:
        break

cap.release()
cv2.destroyAllWindows()