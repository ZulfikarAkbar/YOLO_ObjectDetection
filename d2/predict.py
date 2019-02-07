from darkflow.net.build import TFNet
import cv2

from io import BytesIO
import time
import requests
from PIL import Image
import numpy as np

options = {"model": "cfg/tiny-yolo-voc-1c.cfg", "load": "bin/tiny-yolo-voc.weights", "gpu":1.0, "threshold": 0.1}

tfnet = TFNet(options)

ulatSeen = 0
def handleUlat():
    pass

while True:
    r = requests.get('http://192.168.43.98:8080/image.jpg') # replace with your ip address
    curr_img = Image.open(BytesIO(r.content))
    curr_img_cv2 = cv2.cvtColor(np.array(curr_img), cv2.COLOR_RGB2BGR)

    # uncomment below to try your own image
    #imgcv = cv2.imread('./sample/bird.png')
    result = tfnet.return_predict(curr_img_cv2)
    #print(result)
    
    for detection in result:
        if(detection['label'] == 'Pest! (nettle_caterpillar)'):
            print("Pest detected!")
            ulatSeen += 1
            curr_img.save('save_ulat_api/%i.jpg' % ulatSeen)
        
    print('running again')
    time.sleep(4)
