from darkflow.net.build import TFNet
import cv2

from io import BytesIO
import time
import requests
from PIL import Image, ImageDraw
import numpy as np


import glob

options = {"model": "cfg/tiny-yolo-voc-2c.cfg", "load": "bin/tiny-yolo-voc.weights", "gpu":1.0, "threshold": 0.01}

tfnet = TFNet(options)


counter = 0
for filename in glob.glob('save_ulat_api/*.jpg'):
    counter += 1
    curr_img = Image.open(filename).convert('RGB')
    curr_imgcv2 = cv2.cvtColor(np.array(curr_img), cv2.COLOR_RGB2BGR)

    result = tfnet.return_predict(curr_imgcv2)
    #print(result[0]['confidence'])
    #print(type(result))
    res = [x['confidence'] for x in result]
    #print(res)
    best_index = np.argsort(res)
    best_res = [result[i]['confidence'] for i in best_index[-4:]]
    print(best_res)

    draw = ImageDraw.Draw(curr_img)
    #print(type(best_index))
    result = [result[i] for i in best_index[-4:]]
    # index=best_index[-4:]
    # print(index)
    # result = result[index]
    print(result)
    #print(result)
    for det in result:
        draw.rectangle([det['topleft']['x'], det['topleft']['y'], 
                        det['bottomright']['x'], det['bottomright']['y']],
                       outline=(255, 0, 0))
        draw.text([det['topleft']['x'], det['topleft']['y'] - 13], det['label'], fill=(255, 0, 0))
    # draw.rectangle([result['topleft']['x'], result['topleft']['y'], 
    #                 result['bottomright']['x'], result['bottomright']['y']],
    #                 outline=(255, 0, 0))
    # draw.text([result['topleft']['x'], result['topleft']['y'] - 13], result['label'], fill=(255, 0, 0))
    curr_img.save('ulat_ke/%i.jpg' % counter)
    

