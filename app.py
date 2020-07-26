import numpy as np
from cv2 import cv2
from numpy.core.defchararray import less
from numpy.core.fromnumeric import size

# Loading Yolo
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
image = 'Image.jpg'

classes = []
with open('coco.names', 'r') as file:
    classes = [lines.strip() for lines in file.readlines()]

layer_names = net.getLayerNames()
output_layers = [(layer_names[i[0]-1]) for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))
# Loading image
img = cv2.imread(image)
img = cv2.resize(img, None, fx=0.3, fy=0.4)
width, height, channels = img.shape
# print(img.shape)
# detecting object
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0,0,0), True, crop=False)
# scale factor, size, mean & swapRB, RGB=True
for b in blob:
    for n, img_blob in enumerate(b):
        cv2.imshow(str(n), img_blob)

net.setInput(blob)
outs = net.forward(output_layers)
# print(outs)
boxes = []
confidences = []
class_ids = []
# displaying info on screen
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5 :
            # object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2)
            # circle args(center, radius, color/green, thickness)
            # rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

font = cv2.FONT_HERSHEY_PLAIN
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
# boxes, scores, score threshold, nms threshold

num_of_Obj_detected = len(boxes)
for i in range(num_of_Obj_detected):
    if i in indexes:
        x, y, w, h = boxes[i]
        labels = str(classes[class_ids[i]])
        # print(labels)
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, labels, (x, y + 35), font, 2, color, 2)
        # image, text, origination, fontface, fontscale, color, thickness

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

