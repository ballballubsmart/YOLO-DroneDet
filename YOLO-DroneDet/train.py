import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    p = 'yolov8n+MDC+MSH.yaml'
    model = YOLO(p)
    # model.load('yolov3-tiny.pt') # loading pretrain weights
    model.train(
                task='detect',
                data='Drone.yaml',
                cache='disk',
                imgsz=640,
                epochs=600,
                batch=32,
                workers=8,
                device='0',
                optimizer='SGD',  # using SGD
                patience=100,  # (int) epochs to wait for no observable improvement for early stopping of training
                # resume='', # last.pt path
                amp=False,  # close amp
                project='runs/train/',
                name='yolov8',


                )