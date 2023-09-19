import pathlib, os

from phaser.utils import ImageLoader as IL
from phaser.hashing import PHASH

print("Running script.")
script_dir = f"{os.sep}".join(os.path.abspath(__file__).split(os.sep)[:-1])
print(script_dir)

IMGPATH = os.path.join(script_dir,'images')

list_of_images = [str(i) for i in pathlib.Path(IMGPATH).glob('**/*')]

_img = IL(path=list_of_images[0])

print(PHASH().fit(_img.image))