import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/model_improve/yolov8-600.yaml/weights/best.pt') # select your model.pt path
    model.predict(source='D:\\pycharmdata\\datasets\\dataset622\\dataset1k\\detect',
                  imgsz=640,
                  project='runs/test',
                  name='test',
                  save=True,
                  save_txt=True,
                  conf=0.5,
                  # visualize=True # visualize model features maps
                )