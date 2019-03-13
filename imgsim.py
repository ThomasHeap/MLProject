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
                    help=('Path to the generated images'))
parser.add_argument('--dataPath', type=str,
                    help=('Path to the dataset images'))
parser.add_argument('--savePath', type=str,
                    help=('Path to the save images'))
args = parser.parse_args()



targetPath = pathlib.Path(args.targetPath)
dataPath = pathlib.Path(args.dataPath)

targetFiles = list(targetPath.glob('*.jpg')) + list(targetPath.glob('*.png'))
dataFiles = list(dataPath.glob('*.jpg')) + list(dataPath.glob('*.png'))

def calculateDistance(i1, i2):
    return np.sum((i1-i2)**2)


bestImages = []
worstImages = []
for g in targetFiles:
    targetImg = imageio.imread(str(g))
    distances = []
    for f in dataFiles:
        dataImg = transform.resize(imageio.imread(str(f)), (128,128,3)).astype(np.float32)
        distances.append([f, calculateDistances(targetImg, dataImg)])

    distances.sort(key=lambda x: x[1])
    bestImages.append([g, distances[:5]])
    worstImages.append([g, distances[-5:]])



bestImages.sort(key=lambda x: sum(x[1][:][1]))
worstImages.sort(key=lambda x: -sum(x[1][:][1]))

for i in range(5):
    worstIm = worstImages[i][0]
    bestIm = bestImages[i][0]
    worstIm.save('worst_sample{0}.png'.format(i))
    bestIm.save('best_sample{0}.png'.format(i))
    for j in range(5):
        im1 = Image.open(bestImages[1][i][0])
        im2 = Image.open(worstImages[1][i][0])
        im1.save('{0}/best_nn{1}{2}.png'.format(args.savePath ,i,j) )
        im2.save('{0}/worst_nn{1}{2}.png'.format(args.savePath ,i,j) )
