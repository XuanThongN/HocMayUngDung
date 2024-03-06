import os
from imageai.Detection import ObjectDetection

folder = "D:/1. PXU/16. Hoc may va khoa hoc du lieu/3.HocMay01/mo-hinh/luu-tru/yolov3.pt"
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(folder)
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="D:/1. PXU/16. Hoc may va khoa hoc du lieu/2. API/ResfulAPI/images/car-image.png",
                                             output_image_path="D:/1. PXU/16. Hoc may va khoa hoc du lieu/2. API/ResfulAPI/images/output-car-image.png",
                                             minimum_percentage_probability=30)
for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
    print("--------------------------------")


