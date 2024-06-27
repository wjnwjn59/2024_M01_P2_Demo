import os
import uuid
import gdown
import zipfile

from ultralytics import YOLOv10
from config.model_config import Detector_Config

def export_model(origin_model_path, new_model_path):
    model = YOLOv10(origin_model_path)

    model.export(new_model_path)

def download_model():
    url = 'https://drive.google.com/file/d/1qMkeShHvvp5zix5OGWcxcav4YPOCqu_c/view?usp=drive_link'
    output_file = 'models/yolov10/weights/model_weight.zip'
    unzip_dest = 'models/yolov10/weights'
    os.makedirs(unzip_dest, exist_ok=True)

    gdown.download(url, 
                   output_file, 
                   quiet=True,
                   fuzzy=True)

    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(unzip_dest)

    export_model(origin_model_path=Detector_Config.origin_weight_path,
                 new_model_path=Detector_Config.export_weight_path)


def generate_name():
    uuid_str = str(uuid.uuid4())

    return uuid_str

def save_upload_file(upload_file, save_folder='images'):
    os.makedirs(save_folder, exist_ok=True)
    if upload_file:
        new_filename = generate_name()
        save_path = os.path.join(save_folder, new_filename)
        with open(save_path, 'wb+') as f:
            data = upload_file.read()
            f.write(data)

        return save_path
    else:
        raise('Image not found.')
    
def delete_file(file_path):
    os.remove(file_path)