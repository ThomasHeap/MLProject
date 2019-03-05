import argparse
import shutil
import os
parser = argparse.ArgumentParser()
parser.add_argument('--data', required = True, help='')
parser.add_argument('--image_loc', required = True)
parser.add_argument('--output', required = True)
parser.add_argument('--genre', help='')
parser.add_argument('--style', help='')
opt = parser.parse_args()

ids = []

with open(opt.data) as f:
    for line in f:
        line = line.split(',')
        #unpack object
        genre = line[4]
        style = line[3]
        filename = line[0]

        #when both genre and style are entered
        if(opt.genre and opt.style):
            if(opt.genre.lower() == genre.lower() and opt.style.lower() == style.lower()):
                ids.append(filename)
        elif(opt.genre and opt.genre.lower() == genre.lower()):
            ids.append(filename)
        elif(opt.style and opt.style.lower() == style.lower()):
            ids.append(filename)

for i in ids:
    shutil.copy('{0}/{1}'.format(opt.image_loc, i), '{0}'.format(opt.output))	
