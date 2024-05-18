import os
import yaml

yaml_path = 'human_data.yaml'
label_lst = ['Human']
data_dict = {
    'path': os.getcwd(),
    'train': 'train/images',
    'val': 'val/images',
    'nc': len(label_lst),
    'names': label_lst
}

with open(yaml_path, 'w') as f:
    yaml.dump(data_dict, f, sort_keys=False)