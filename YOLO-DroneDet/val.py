import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'YOLOv8n+MDC+MSH+LAMP+LOG.pt')
    model.val(data='./Drone.yaml',
              split='test',
              imgsz=640,
              batch=32,
              workers=8,
              # rect=False,
              save=False,
              # save_txt=True,
              save_json=True,# if you need to cal coco metrice
              project='runs/',
              name='YOLOv8n+MDC+MSH+LAMP+LOG',
              # iou=0.5
              )