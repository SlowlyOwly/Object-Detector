import os
import cv2
import matplotlib.pyplot as plt
import util
import numpy as np


# define constants
model_cfg_path = os.path.join('.', 'model', 'cfg', 'yolov3.cfg')
model_weights_path = os.path.join('.', 'model', 'weights', 'yolov3.weights')
class_names_path = os.path.join('.', 'model', 'class.names')

img_path = './pexels-diana-huggins-615369.jpg'

# load class names
with open(class_names_path, 'r') as f:
    class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
    f.close()

# load model
net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

# load image

img = cv2.imread(img_path)

H, W, _ = img.shape

# convert image
blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), True)

# get detection

net.setInput(blob)

detections = util.get_outputs(net)

# bboxes, class_ids, confidences

bboxes = []
class_ids = []
confidences = []

for item in detections:
    bbox = item[:4]

    xc, yc, w, h = bbox
    bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

    bbox_confidence = item[4]

    class_id = np.argmax(item[5:])
    confidence = np.amax(item[5:])

    bboxes.append(bbox)
    class_ids.append(class_id)
    confidences.append(confidence)

# apply nms

bboxes, class_ids, confidences = util.NMS(bboxes, class_ids, confidences)

print(len(bboxes))

# plot

for bbox_, bbox in enumerate(bboxes):
    xc, yc, w, h = bbox

    cv2.putText(img,
                class_names[class_ids[bbox_]],
                (int(xc - (w / 2)), int(yc + (h / 2) - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                5,
                (0, 255, 0),
                5)

    cv2.rectangle(img,
                  (int(xc - (w / 2)), int(yc - (h / 2))),
                  (int(xc + (w / 2)), int(yc + (h / 2))),
                  (0, 255, 0),
                  10)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
