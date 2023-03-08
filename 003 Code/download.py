import gdown
import os
from os.path import exists

# Download zip file from gdrive

if not exists("./data/image.zip"):
    google_path = 'https://drive.google.com/uc?id='
    file_id = '1E4JBTFGuGiGVjjcBDU9091Ahg480StjT'
    output_name = './data/image.zip'
    gdown.download(google_path+file_id,output_name,quiet=False)


# unzip 
if not exists("./data/image"):
    os.makedirs("./data/image", exist_ok=True)
    print("unzip")
    os.system("unzip -uq ./data/image.zip -d ./data/image/")
    os.system("rm ./data/image.zip")


