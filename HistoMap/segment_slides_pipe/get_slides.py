import slideio
import matplotlib.pyplot as plt
import random
import cv2 as cv
import numpy as np
from PIL import Image
import sys
import os
import shutil

### reads in slide name
which_slide = sys.argv[1]
wd = '/vast/palmer/home.grace/kmf69/segment_slides_pipe/' + which_slide

os.chdir(wd)
# make directories for saving things
if os.path.exists(wd + '/raw_images'):
	shutil.rmtree(wd + '/raw_images')
if os.path.exists(wd + '/csv_files'):
	shutil.rmtree(wd + '/csv_files')
if os.path.exists(wd + '/crop_images'):
	shutil.rmtree(wd + '/crop_images')
if os.path.exists(wd + '/stitch_images'):
	shutil.rmtree(wd + '/stitch_images')


os.mkdir("raw_images")
os.mkdir("csv_files")
os.mkdir("crop_images")
os.mkdir("stitch_images")


img_path = wd + '/' + which_slide + '.svs'

slide = slideio.open_slide(img_path, driver = 'SVS') #Read in slide
scene = slide.get_scene(0) #Change to scene
block = scene.read_block(size=(500,0)) #Get block

total_height = scene.size[0]
total_width = scene.size[1]

### save width and height dimensions for future code
print("Now making 512x512 images...")
os.chdir(wd)


block_width = 512
block_height = 512

num_blocks_y = round(total_width/block_width)
num_blocks_x = round(total_height/block_height)

image_dir = wd + '/image_dims.txt'
with open(image_dir,'w') as f:
	f.write(wd + '\n')
	f.write(str(num_blocks_x) + '\n')
	f.write(str(num_blocks_y) + '\n')

print(total_width, total_height)
print(num_blocks_x,num_blocks_y)
#sys.exit()

x_values = [0]
y_values = [0]

#Get x values for cutting into smaller imagess
for index in range(num_blocks_x): 
	x_values.append(x_values[index] + block_width)
for index in range(num_blocks_y):
	y_values.append(y_values[index] + block_height)

image_sample = []



#x_sample[1]
save_dir = wd + '/raw_images'
for i in range(len(x_values)):
	print(i)
	for j in range(len(y_values)):
		image_sample = scene.read_block((x_values[i], y_values[j], block_width, block_height))
		name = '/image (' + str(i) + ',' + str(j) + ').png'
		name = save_dir + name
		plt.imsave(name, image_sample)

