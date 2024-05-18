import os
import uuid
import gdown
import zipfile

def download_model():
    url = 'https://drive.google.com/file/d/1jl4TmG5l9ETpTs3sH4BHyLayXWRMc7E1/view?usp=sharing'
    output_file = 'models/yolov9/weights/model_weight.zip'
    unzip_dest = 'models/yolov9/weights'
    os.makedirs(unzip_dest, exist_ok=True)

    gdown.download(url, 
                   output_file, 
                   quiet=True)

    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(unzip_dest)

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