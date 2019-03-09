import sys
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from scipy import linalg
from scipy.misc import imread
from skimage import transform
import imageio
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = sys.maxsize
from matplotlib.image import imread
from matplotlib.pyplot import imshow

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--targetPath', type=str,
                    help=('Path to the target generated image'))
parser.add_argument('--dataPath', type=str,
                    help=('Path to the dataset images'))
args = parser.parse_args()

targetImg = imread(args.targetPath)

dataPath = pathlib.Path(args.dataPath)

files = list(dataPath.glob('*.jpg')) + list(dataPath.glob('*.png'))

def calculateDistance(i1, i2):
    return np.sum((i1-i2)**2)

distances = []

for f in files:
    dataImg = transform.resize(imageio.imread(str(f)), (128,128,3)).astype(np.float32)
    distances.append([calculateDistance(targetImg, dataImg), f])
    print((len(distances) / len(files)) * 100)


distances.sort(key=lambda x: x[0])

print(distances[:5])
imtarg = Image.open(args.targetPath)
imtarg.show()
imtarg.save('target.png')
for i in range(5):
    im = Image.open(distances[i][1])
    im.show()
    im.save('best{0}.png'.format(i))
