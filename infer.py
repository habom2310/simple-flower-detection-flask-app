import cv2
import keras
from PIL import Image
import base64
import io
import numpy as np


class Inference:
    def __init__(self):
        self.model = keras.models.load_model("best.hdf5")
        # self.labels = []
        with open("label.txt") as f:
            self.labels = f.readlines()

    def process_image(self, img):
        if (img.shape[0] != 224 or img.shape[1] != 224):
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)

        img = (img/127.5)
        img = img - 1
        img = np.expand_dims(img, axis=0)
        return img

    def base64_to_img(self, base64_string):
        # print(base64_string)
        imgdata = base64.b64decode(str(base64_string))
        image = Image.open(io.BytesIO(imgdata))
        return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    def predict(self, base64_string):
        img = self.base64_to_img(base64_string)
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        pred = self.model.predict(self.process_image(img))
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        print(np.argmax(pred))
        # print(pred)
        result = self.labels[np.argmax(pred)]
        conf = pred.max()
        return result, conf