import os
from PIL import Image
import PIL
import numpy as np
import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

## get text file 
which_slide = sys.argv[1]
#txt_file = sys.argv[2]

# parser = argparse.ArgumentParser(description = 'text file with stuff')
# parser.add_argument('strings', metavar = 'S')

# args = parser.parse_args()

# test_file = args.strings
wd = '/vast/palmer/home.grace/kmf69/segment_slides_pipe/'


# with open(txt_file) as f:
# 	lines = [line.rstrip() for line in f]
# 	wd = lines[0]

image_dir = wd + which_slide+ "/raw_images"

def makeCollage(image_dir, x_dim,y_dim):
	"""Merge two images into one, displayed side by side
	:param file1: path to first image file
	:param file2: path to second image file
	:return: the merged Image object
	"""
	max_x = 0
	max_y = 0

	all_files = os.listdir(image_dir)
	all_files = [file for file in all_files if '.nfs' not in file]
	# find the dimensions of the image
	print("Now getting dimensions...")
	for file in all_files:
		filename = os.fsdecode(file)

		coord = filename[5:-4]
		#print(coord)
		#sys.exit()
		coord = coord.replace(" ", "")
		coord = coord.replace("(","")
		coord = coord.replace(")","")

		x = float(coord.split(',')[0])
		y = float(coord.split(',')[1])
		#coord = coord[-4:]

		if x > max_x:
			max_x = x
		if y > max_y:
			max_y = y
	max_x = max_x + 1
	max_y = max_y + 1
	result_width = int(max_x * x_dim)
	result_height = int(max_y * y_dim)

	# result.save()
	result = Image.new('RGB', (result_width, result_height))
	for file in all_files:
		filename = os.fsdecode(file)

		coord = filename[5:-4]
		coord = coord.replace(" ", "")
		coord = coord.replace("(","")
		coord = coord.replace(")","")

		x = float(coord.split(',')[0])
		y = float(coord.split(',')[1])

		image1 = PIL.Image.open(image_dir +'/' +filename)
		result.paste(im=image1, box=(int(x*x_dim), int(y*y_dim)))
	#result.save("../final_collage-uncompressed.png")
	base_width = 2500
	#image = Image.open('example-image.jpg')
	width_percent = (base_width / float(result.size[0]))
	hsize = int((float(result.size[1]) * float(width_percent)))
	result = result.resize((base_width, hsize), PIL.Image.LANCZOS)
	
	os.chdir(image_dir)
	result.save("../combined_total.png")
	return result

def graphNucs(df, orig_image, save_dir , i , j, offset, x_dim = 512, y_dim = 512):
	# takes in csv file of nuclei coords and image and graphs them

		#print(temp_df)
	#input('...')
	contours = df['Contours'].to_numpy()
	mpl.rcParams['figure.dpi'] = 300
	contour_num = 0
	for contour in contours:
		#k = k + 1
		#print(k)
		temp_cont = contour[1:-1]
		temp_cont  = temp_cont.splitlines()
		temp_cont  = [x.strip() for x in temp_cont if x]
		temp_cont = [x.replace('[','') for x in temp_cont]
		temp_cont = [x.replace(']','') for x in temp_cont]
		temp_cont  = [x.strip() for x in temp_cont if x]
		temp_cont = [x.replace('   ',' ') for x in temp_cont]
		temp_cont = [x.replace('  ',' ') for x in temp_cont]
		#print(temp_cont)
		#input('...')
		x_list = []
		y_list = []
		#print(temp_cont)

		k = 0 
		print("Now plotting contour " + str(contour_num))
		contour_num = contour_num + 1
		#k = k + 1

		first_x = 0
		first_y = 0
		#point_list = []
		
		if '...' in temp_cont:
			continue
		else:
			x_offset = 0
			y_offset = 0
			if offset:
				x_offset = x_dim * i
				y_offset = y_dim * j

			for coord in temp_cont:
				#print(coord)
				coord = ' '.join(coord.split())
				#print(coord)
				#print(coord)
				x = float(coord.split(' ')[0].replace(',','')) - x_offset
				y = float(coord.split(' ')[1].replace(',','')) - y_offset

				#x, y = float(x), float(y)
				x_list.append(x)
				y_list.append(y)
				if k == 0: 
					first_x = x
					first_y = y
					k = 1

			x_list.append(first_x)
			y_list.append(first_y)

		#print("x: ", x_list)
		#print("y: ",y_list)

		plt.plot(x_list,y_list,'yellow',linewidth = 0.3)
	#cv2.drawContours(image1, temp_df['Contours'], -1, color=(0, 255, 255), thickness=2)
	plt.imshow(orig_image, cmap = 'Greys', alpha = 0.6)
	#plt.show()
	
	plt.savefig(save_dir)
	plt.close()

x_dim = 512 
y_dim = 512

total_i = 0
total_j = 0

#save_dir = "/vast/palmer/home.grace/rn367/HistoMap/raw_images/final_trace.png"

#csv_dir = "C:/Users/rn367/Desktop/HistoMap/ryan_images/csv_files"
#csv_title = csv_dir + '/combined_total.csv'
#total_df = pd.read_csv(csv_title)
result = makeCollage(image_dir, x_dim, y_dim)

#graphNucs(total_df, result, save_dir, total_i, total_j, False )


