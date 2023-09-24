from moviepy.editor import VideoFileClip
import numpy as np
import os
from datetime import timedelta
from imageai.Detection import ObjectDetection
import torch

def format_timedelta(td):
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")

# Извлечение кадров из видео. Кадры сохраняются как изображения jpg
def frames(video_file):
    os.chdir(PATH)
    video_clip = VideoFileClip(video_file)
    filename, _ = os.path.splitext(video_file)
    filename += "-frames"
    if not os.path.isdir(filename):
        os.mkdir(filename)
    saving_frames_per_second = min(video_clip.fps, SAVING_FRAMES_PER_SECOND)
    step = 1 / video_clip.fps if saving_frames_per_second == 0 else 1 / saving_frames_per_second
    for current_duration in np.arange(0, video_clip.duration, step):
        frame_duration_formatted = format_timedelta(timedelta(seconds=current_duration)).replace(":", "-")
        frame_filename = os.path.join(filename, f"frame{frame_duration_formatted}.jpg")
        video_clip.save_frame(frame_filename, current_duration)
    os.chdir("..")
    return filename

# Распознавание объектов в кадре
def detect(frames_dir):
    os.chdir(PATH+"/"+frames_dir)
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "../../retinanet_resnet50_fpn_coco-eeacb38b.pth"))
    detector.loadModel()
    
    custom_objects = detector.CustomObjects(cell_phone=True, person=True) # Распознаем людей и телефоны

    f=open(frames_dir+".log", "w") # логи

    for frame in os.listdir():
        if frame==frames_dir+".log": # костыль :)
            continue
        cell_phone=[]
        person=[]
        detections = detector.detectObjectsFromImage(
            custom_objects=custom_objects, 
            input_image=os.path.join(execution_path , frame), 
            output_image_path=os.path.join(execution_path , "result-"+frame), 
            minimum_percentage_probability=30)
        
        for eachObject in detections:
            print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
            # координаты объектов распределяем по массивам
            if eachObject["name"]=="cell phone":
                cell_phone.append({"type":"cell phone", "points":eachObject["box_points"]})
            else:
                person.append({"type":"person", "points":eachObject["box_points"]})
            print("--------------------------------")
        print(cell_phone, person)
        alert=check_width(cell_phone, person) # Расчет расстояний
        print(alert)
        print(frames_dir, frame, cell_phone, person, alert, file=f)
        if alert==True: # Если обнаружен факт использования телефона - анализ остальных фреймов не требуется
            break
    os.chdir("../../")
    f.close()


# Расчет расстояний. Ищем центр телефона, затем проверяем находится ли он в квадрате сотрудника. Если да - это инцидент, возвращаем True
def check_width(cell_phone, person):
    for c in cell_phone:
        cx, cy=box_corner_to_center(c["points"])
        for p in person:
            x1, y1, x2, y2 = p["points"][0], p["points"][1], p["points"][2], p["points"][3]
            if cx>x1 and cx<x2 and cy>y1 and cy<y2: 
                return True
    return False


def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[0], boxes[1], boxes[2], boxes[3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy

MODEL="./retinanet_resnet50_fpn_coco-eeacb38b.pth" # Путь к модели
PATH="./videos" # Директория с видео
SAVING_FRAMES_PER_SECOND=0.02 # количество кадров, выделяемое из одной секунды видео. Может быть дробным

if __name__ == "__main__":
    for video_file in os.listdir(PATH):
        print("FILE: "+video_file)
        frames_dir=frames(video_file) # извлекаем кадры из видео
        detect(frames_dir) # Распознаем объекты на всех кадрах