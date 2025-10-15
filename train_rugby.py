from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")

yaml_path = "dataset.yaml"
if not os.path.exists(yaml_path):
    raise FileNotFoundError(f"No se encontró {yaml_path}. Verifica la ruta.")

print("Iniciando entrenamiento del modelo YOLOv8 para Rugby...")

model.train(
    data=yaml_path,   
    epochs=40,        
    imgsz=640,        
    batch=16,         
    name="rugby_train",  
    project="runs/train" 
)

print("Entrenamiento finalizado. Pesos guardados en runs/train/rugby_train#/weights/best.pt")
