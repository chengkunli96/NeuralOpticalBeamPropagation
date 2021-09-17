"""
get all images' path from Face-lfw dataset, and save as a file.
"""

import os

def is_image_file(filename):
    flag = any(
        filename.endswith(extension) for extension
        in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    )
    return flag

# get image paths
def getImagePathes(path):
    all = os.walk(path)
    img_pths = []
    for path, dir, filelist in all:
        for filename in filelist:
            if is_image_file(filename):
                img_pths.append(os.path.join(path, filename))
    return img_pths
