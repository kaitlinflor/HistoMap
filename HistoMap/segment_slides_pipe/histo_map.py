from stardist.models import StarDist2D
from stardist.data import test_image_he_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import matplotlib as mpl
import pandas as pd
import numpy as np
from read_roi import read_roi_zip
import os
import math
import sys
from shapely.geometry import Polygon
import os
import glob
import pandas as pd


def trimNuclei(wd, main_image_coord, orig_image, df, i,j, crop_size , x_dim = 512,y_dim = 512 ):
	# get rid of nuclei at the edge so you can replace them with merged region nuclei
	idx_list = []
	x_offset = x_dim * i
	y_offset = y_dim * j
	for idx in range(0,len(df['Contours'])):
		cut_out = False
		for coord_list in df['Contours'][idx]:
			x, y = float(coord_list[0]), float(coord_list[1])


			x = x - x_offset
			y = y - y_offset

			# check if values are close to boundaries
			if i == 0 and j == 0 and ((0 <=x <=crop_size) or (0 <=y <=crop_size) ): 
				cut_out = False
			# elif i == 0 and j == 0 and (((y_dim - crop_size) <=y <= y_dim) or ((x_dim - crop_size) <=x <= x_dim)): 
			# 	cut_out = True
			elif  i != 0 and j == 0 and (0 <=y <=crop_size):
				cut_out = False 
			# elif  i != 0 and j == 0 and (((y_dim - crop_size) <=y <= y_dim) or (((x_dim - crop_size) <=x <= x_dim))):
			# 	cut_out = True
			elif  i == 0 and j != 0 and (0 <=x <=crop_size):
				cut_out = False
			elif 0 <=x <=crop_size  or (x_dim - crop_size) <=x <=x_dim or 0 <=y <=crop_size  or (y_dim - crop_size) <=y <=y_dim :
				cut_out = True
				#break
		if cut_out:
			idx_list.append(idx)
	#print(idx_list)
	#sys.exit()
	#temp_df = df.drop(index = idx_list)
	temp_df = df
	filter_title = wd + "/csv_files/coords_" + main_image_coord + 'filtered.csv'
	temp_df.to_csv(filter_title)
	temp_df = None

	#print(filter_title)
	#input('...')
	temp_df = pd.read_csv(filter_title, sep = ',')
	save_dir = wd + '/traces/image' + main_image_coord + '_trace_filtered.png'
	#graphNucs(temp_df, orig_image, save_dir, i, j, True)

	return temp_df

def contour2Poly(contour):
	# turn contour into polygon
	contour1 = contour
	contour1 = contour1[1:-1]
	contour1  = contour1.splitlines()
	contour1  = [x.strip() for x in contour1 if x]
	contour1 = [x.replace('[','') for x in contour1]
	contour1 = [x.replace(']','') for x in contour1]
	contour1  = [x.strip() for x in contour1 if x]
	contour1 = [x.replace('   ',' ') for x in contour1]
	contour1 = [x.replace('  ',' ') for x in contour1]

	coord_list = []
	for coord in contour1:
		#print(coord)
		if '...' in coord:
			continue
		else:
			coord = ' '.join(coord.split())
			x = float(coord.split(' ')[0].replace(',','')) 
			y = float(coord.split(' ')[1].replace(',','')) 

			coord_list.append((x,y))
	return Polygon(coord_list)

def getDistance(x1, y1, x2, y2):
	return ((((x2 - x1 )**2) + ((y2-y1)**2) )**0.5)
def trimNuclei2(df):
	# get rid of nuclei that are double counted because of boundary effects
	contours = df['Contours'].to_numpy()
	#mpl.rcParams['figure.dpi'] = 300
	idx_list = []
	for i in range(0,len(contours)-1):
		
		contour1 = contours[i]
		poly1 = contour2Poly(contour1)
		poly1_centroid = poly1.centroid 

		#print(poly1_centroid.x)

		for j in range(i+1,len(contours)) : 
			contour2 = contours[j]
			poly2 = contour2Poly(contour2)
			poly2_centroid = poly2.centroid
			#print(poly2_centroid)
			#sys.exit()
			dist = getDistance(poly1_centroid.x, poly1_centroid.y, poly2_centroid.x, poly2_centroid.y )
			# check if poly1 and poly2 intersect and one is 
			if dist < 13 and poly1.intersects(poly2): 
				#print(dist)
				if poly1.area < poly2.area:
					idx_list.append(i)
				elif poly1.area > poly2.area:
					idx_list.append(j)
		#print(dist, idx_list)
	idx_list = list(set(idx_list))
		#sys.exit()
	temp_df = df.drop(index = idx_list)
	
	return temp_df
	#sys.exit()

def graphNucs(df, orig_image, save_dir , i , j, offset, x_dim = 512, y_dim = 512):
	# takes in csv file of nuclei coords and image and graphs them

		#print(temp_df)
	#input('...')
	contours = df['Contours'].to_numpy()
	mpl.rcParams['figure.dpi'] = 300
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

		#plt.plot(x_list,y_list,'yellow',linewidth = 0.6)
	#cv2.drawContours(image1, temp_df['Contours'], -1, color=(0, 255, 255), thickness=2)
	#plt.imshow(orig_image, cmap = 'Greys', alpha = 0.6)
	#plt.show()
	
	plt.savefig(save_dir)
	plt.close()

def merge_images(image1_array, image1_dir, image2_array, image2_dir, stitch_direction, i,j, one_image_size = 512):
	"""Merge two images into one, displayed side by side
	:param file1: path to first image file
	:param file2: path to second image file
	:return: the merged Image object
	"""
	#image1.show()
	#image2.show()

	width1, height1, c1 = image1_array.shape
	width2, height2, c2 = image2_array.shape

	image1 = Image.open(image1_dir)
	image2 = Image.open(image2_dir)

	#print(width1 + width2)
	if stitch_direction == 'up':
		result_height = height1 + height2
		result_width = max(width1, width2)
		result = Image.new('RGB', (result_width, result_height))
		result.paste(im=image2, box=(0, 0))
		result.paste(im=image1, box=(0, height1))

		return result
	if stitch_direction == 'down':
		result_height = height1 + height2
		result_width = max(width1, width2)
		result = Image.new('RGB', (result_width, result_height))
		result.paste(im=image1, box=(0, 0))
		result.paste(im=image2, box=(0, height1))

		return result
	if stitch_direction == 'left':
		result_width = width1 + width2
		result_height = max(height1, height2)
		result = Image.new('RGB', (result_width, result_height))
		offset = 0
		if result_height > one_image_size and j != 0:
			offset = one_image_size
		result.paste(im=image2, box=(0, 0 + offset))
		result.paste(im=image1, box=(width1, 0))

		return result
	if stitch_direction == 'right':
		result_width = width1 + width2
		result_height = max(height1, height2)
		result = Image.new('RGB', (result_width, result_height))
		offset = 0
		if result_height > one_image_size and j != 0:
			offset = one_image_size
		result.paste(im=image1, box=(0, 0))
		result.paste(im=image2, box=(width1, 0 + offset))

		return result
	
def open_file(wd, file_list, coord):
	image1 = ''
	#print(coord)

	#sys.exit()
	for image in file_list:
		#print(image, coord in image)
		if coord in image:
			#print(image)
			image_dir = wd + '/' + image

			image1 = cv2.imread(image_dir)
	#print(image1.size)
	#print(image1.shape)

	return image1, image_dir


def get_Nucs(img, threshold, i, j, wd, main_image_coord, crop_dir = 'None', cropped= False,  crop_size = 80, x_dim = 512, y_dim = 512):
	### -------------------------------------------- ###
	# This method traces out all nuclei in image and 
	# returns contours and coords of nuclei
	### -------------------------------------------- ###

	#img = cv2.imread(img_dir)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img1 = img.copy()

	# filter out all white images
	#print(img)
	if np.mean(img) > 200 or np.mean(img) < 20:
		print('All white')
		return "white", "white"
	else:
		print('Not all white')

	mpl.rcParams['figure.dpi'] = 500

	labels, polygons = model.predict_instances(normalize(img), prob_thresh= threshold)

	#cv2.imwrite('labels.jpg', labels)
	# img_b = cv2.imread('labels.jpg')

	# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
	# ax1.axis("off")
	# ax1.imshow(img)
	# ax2.axis("off")
	# ax2.imshow(render_label(labels, img=img))
	# ax3.axis("off")
	# ax3.imshow(img_b)

	#plt.show()
	#cv2.imwrite(wd + '/bw/image' + main_image_coord + '_bw.png', img_b)


	## Get coords, points and probability
	coords = list(polygons.values())[0]
	points = list(polygons.values())[1]
	prob = list(polygons.values())[2]


	mpl.rcParams['figure.dpi'] = 100
	## Get real contours, plot coordinates
	new_coords_plot = []
	new_coords_coords = []
	x_offset = 0
	y_offset = 0
	if cropped:
		# for some reason the x and y's are switched
		if crop_dir == 'up' or crop_dir == 'down':
			x_offset = x_dim * i 
			y_offset = y_dim * j  - crop_size
		if crop_dir == 'left' or crop_dir == 'right':
			x_offset = x_dim * i - crop_size
			y_offset = y_dim * j
	else: 
		x_offset = x_dim * i
		y_offset = y_dim * j
	for k, coord in enumerate(coords):
		zipped_plot = zip(coord[0] , coord[1] )
		zipped_coords = zip(coord[0] + y_offset, coord[1] + x_offset)
		xy_plot = []
		xy_coords = []
		for a, b in zipped_plot:
			xy_plot.append([b, a])
			#xy_coords.append([b, a])
		for a, b in zipped_coords:
			xy_coords.append([b, a])
		new_coords_plot.append(xy_plot)
		new_coords_coords.append(xy_coords)

	contours_plot = np.array(new_coords_plot).reshape((-1,1,2)).astype(np.int32)
	contours_coords = np.array(new_coords_coords).reshape((-1,1,2)).astype(np.int32)
	new_coords_coords = np.array(new_coords_coords).astype(np.float32)

	#image3 = img1.copy()
	# cv2.drawContours(image3, contours_plot, -1, color=(0, 255, 255), thickness=2)
	# #plt.imshow(image3, cmap = 'Greys')
	# cv2.imwrite(wd + '/traces/image' + main_image_coord + '_trace.png', image3)
	plt.close()
	return contours_coords, new_coords_coords


def getNucParams(contours, new_coords):
	##Get Aspect Ratio
	aspect_ratio = []
	#print(new_coords)
	#print(len(contours))
	#print(new_coords)
	#sys.exit()
	for coord in new_coords:
		coord = np.round_(coord, decimals = 3)

		#input("...")
		#print(contour) 
		#sys.exit()
		dims = cv2.boundingRect(coord)
		#sys.exit()
		#dims = cv2.boundingRect(coord)
		#print(dims)
		aspect_ratio.append(float(min(dims[2], dims[3])/max(dims[2], dims[3])))

	#print(max(aspect_ratio), min(aspect_ratio))
	#if 
	#aspect_ratio.count(max(aspect_ratio))

	# get Moments
	cX = []
	cY = []
	centers = []

	for i, c in enumerate(new_coords):
		M = cv2.moments(c)
		cX.append(int(M['m10']/ M['m00']))
		cY.append(int(M['m01']/ M['m00']))

	for a, b in zip(cX, cY):
		centers.append([a, b])

	# get area, perimeter, and circularity
	contour_areas = [cv2.contourArea(cont) for cont in new_coords]
	perimeters = [cv2.arcLength(cont, True) for cont in new_coords]

	circularity = []
	for i in range(len(contour_areas)):
		circularity.append((4 * np.pi * contour_areas[i])/ (perimeters[i]**2))


	measure_dict = {'Area' : contour_areas, 'Aspect Ratio': aspect_ratio, 'Centers': centers, 'Perimeter': perimeters, 
	'Circularity': circularity, 'Contours' : tuple(new_coords)}

	df = pd.DataFrame(measure_dict)
	return df
	#csv_data = df.to_csv('data\csv\csv' + str(image_num) + '.csv')

def refineCrop(crop_df, x_dim, y_dim, stitch_direction, crop_range):
	# get rid of cropped image nuclei that are not on the edge so that way you don't double count nuclei
	contours = crop_df['Contours']
	# mpl.rcParams['figure.dpi'] = 300

	idx_list = []
	for idx in range(0,len(contours)):
		#k = k + 1
		#print(k)
		contour = contours[idx]
	
		x_list = []
		y_list = []

		if '...' in contour:
			continue
		else:
			for coord in contour:
				x_offset = x_dim * j
				y_offset = y_dim * i

				x = coord[0] - x_offset #float(coord.split(' ')[0].replace(',','')) - x_offset
				y = coord[1] - y_offset#float(coord.split(' ')[1].replace(',','')) - y_offset
				#print(x,abs(x-x_dim),y,  abs(y-y_dim))
				if stitch_direction == 'up' or stitch_direction == 'down':
					if abs(y-y_dim) > crop_range: #or abs(y-y_dim) > crop_range:
						idx_list.append(idx)
					break
				if stitch_direction == 'left' or stitch_direction == 'right':
					if abs(x-x_dim) > crop_range: #or abs(y-y_dim) > crop_range:
						idx_list.append(idx)
					break

	temp_df = crop_df.drop(index = idx_list)

	return temp_df


os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = StarDist2D.from_pretrained('2D_versatile_he')


# image_num = 1
threshold = 0.5
i_temp = int(sys.argv[1])
which_slide = sys.argv[2]


wd = '/vast/palmer/home.grace/kmf69/segment_slides_pipe/' + which_slide

txt_file = wd + '/image_dims.txt'
with open(txt_file) as f:
	lines = [line.rstrip() for line in f]
	wd = lines[0]
	i_max = int(lines[1])
	j_max = int(lines[2])

step_size = 7
#looping through coordinates
max_i = i_temp + step_size
if i_max - (i_temp + step_size) < 7:
  max_i = i_max

x_dim = 512
y_dim = 512
crop_size = 40

#wd = '/vast/palmer/home.grace/rn367/HistoMap/S12-26040_15'

# read in files
file_list = []
for file in os.listdir(wd + '/raw_images'):
	#print(file)
	if 'DS_Store' not in file:
		#print(type(file))
		file_list.append(file)

# time to stitch 83,128
for i in range(i_temp,max_i):
	for j in range(0,j_max):
		left_image_coord = 'None'
		right_image_coord = 'None'
		up_image_coord = 'None'
		down_image_coord = 'None'

		main_image_coord = '(' + str(i) + ',' + str(j) + ')'
		
		stitch_list = []

		if i == 0 and j == 0:
			down_image_coord = str(i) + ',' + str(j+1)
			right_image_coord = str(i+1) + ',' + str(j)
		elif i > 0 and j == 0:
			left_image_coord = str(i-1) + ',' + str(j)
			down_image_coord = str(i) + ',' + str(j+1)
			if i != i_max - 1:
				right_image_coord = str(i+1) + ',' + str(j)
		elif i == 0 and j > 0:
			up_image_coord = str(i) + ',' + str(j-1)
			right_image_coord = str(i+1) + ',' + str(j)
			if j != j_max - 1:
				down_image_coord = str(i) + ',' + str(j+1)
		elif i == i_max - 1 and j == j_max - 1:
			left_image_coord = str(i-1) + ',' + str(j)
			up_image_coord = str(i) + ',' + str(j-1)
		else:
			left_image_coord = str(i-1) + ',' + str(j)
			up_image_coord = str(i) + ',' + str(j-1)
			if j != j_max - 1:
				down_image_coord = str(i) + ',' + str(j+1)
			if i != j_max - 1:
				right_image_coord = str(i+1) + ',' + str(j)

		#stitch list is up, down, left, right
		stitch_list = [up_image_coord, down_image_coord, left_image_coord, right_image_coord]
		# # don't go up or left to reduce redundant calculations
		stitch_list = ['None', down_image_coord, 'None', right_image_coord]
		stitch_direction = ['up', 'down', 'left', 'right']
		#print(main_image_coord,stitch_list)
		#input('...')

		# get the first image nuclei and then 
		image1, image1_dir = open_file(wd + '/raw_images', file_list, main_image_coord)
		#print(image1_dir)
		#sys.exit()
		first_image = image1.copy()
		print("First image to stitch: ", main_image_coord)
		contours1, new_coords1 = get_Nucs(img = image1, threshold = threshold , i = i, j = j, wd = wd, main_image_coord = main_image_coord,crop_dir = 'None', cropped= False,  crop_size = crop_size, x_dim = x_dim, y_dim = y_dim)
		columns = ["Unnamed", "Area",	"Aspect Ratio", "Centers","Perimeter","Circularity","Contours"]

		title = wd + "/csv_files/coords_" + main_image_coord + 'total.csv'
		if contours1 != 'white':
			image1_df = getNucParams(contours1,new_coords1)
			image1_df = trimNuclei(wd, main_image_coord, image1, image1_df, i,j,crop_size = crop_size )

			#image1_df.columns = columns
			image1_df.to_csv(title,index=False)
		
		#print(image1.size)
		#image1.show()

		#input('...')
		#print(stitch_list)
		#sys.exit()
		# stitch images together sequentially
		for coord_idx in range(0,len(stitch_list)):
			if stitch_list[coord_idx] != 'None':
				print("Second image to stitch: ",stitch_list[coord_idx])
				second_image_coord = '(' + stitch_list[coord_idx] + ')'
				i_coord2 = int(stitch_list[coord_idx].split(',')[0])
				j_coord2 = int(stitch_list[coord_idx].split(',')[1])
				#print(second_image_coord)
				#input('...')
				# get image 2 nuclei 
				image2, image2_dir = open_file(wd + '/raw_images', file_list, second_image_coord)
				
				second_image = image2.copy()
				
				contours2, new_coords2 = get_Nucs(img = image2, threshold = threshold , i = i_coord2, j = j_coord2, wd = wd, main_image_coord = second_image_coord,crop_dir = 'None', cropped= False,  crop_size = crop_size, x_dim = x_dim, y_dim = y_dim)
				
				#print(contours2)
				#sys.exit()
				if contours2 != 'white':
					image2_df = getNucParams(contours2,new_coords2)
					image2_df = trimNuclei(wd, second_image_coord, image2, image2_df, i = i_coord2, j = j_coord2, crop_size = crop_size)
					image2_df.to_csv(title, mode = 'a',index = False, header=False)
					#print("bing")
				temp_image = merge_images(first_image,image1_dir, second_image, image2_dir, stitch_direction[coord_idx], i, j)
				#temp_image.show()

				stitch_title = wd + '/stitch_images/' + main_image_coord + '_' + second_image_coord + '_merge.png'
				temp_image.save(stitch_title)
				crop_image = ''
				
				if stitch_direction[coord_idx] == 'up' or stitch_direction[coord_idx] == 'down':
					left = 0 
					right = x_dim
					top = y_dim - crop_size
					bottom = y_dim + crop_size
					crop_image = temp_image.crop((left, top, right, bottom))
					#crop_image.show()
				if stitch_direction[coord_idx] == 'left' or stitch_direction[coord_idx] == 'right':
					#print(temp_image.size)
					top = 0 
					right = x_dim + crop_size
					bottom = y_dim
					left = x_dim -crop_size
					crop_image = temp_image.crop((left, top, right, bottom))
				# save crop image because life is hard and stupid 
				# save a image using extension
				print("Now getting area in between...")
				if contours2 != 'white':
					crop_title = wd + '/crop_images/' + second_image_coord + '_crop.png'
					
					crop_image.save(crop_title)
	

					crop_image = cv2.imread(crop_title)
					stitch_image = cv2.imread(stitch_title)

					crop_coord = main_image_coord + '_' + stitch_direction[coord_idx] +'_crop'
					crop_i = int(stitch_list[coord_idx].split(',')[0])
					crop_j = int(stitch_list[coord_idx].split(',')[1])

					crop_contours, crop_coords = get_Nucs(img = crop_image, threshold = threshold + 0.1 , i = crop_i, j = crop_j, wd = wd, main_image_coord = crop_coord, crop_dir = stitch_direction[coord_idx], cropped = True, crop_size = crop_size , x_dim = x_dim, y_dim = y_dim)
					#print(crop_contours)
					if crop_contours != 'white' and crop_contours.size != 0:
						#print(crop_contours)				
						crop_df = getNucParams(crop_contours,crop_coords)
						#print(crop_df)
						crop_df = refineCrop(crop_df, x_dim, y_dim, stitch_direction = stitch_direction[coord_idx], crop_range = crop_size-(crop_size*0.05))
						#print(crop_df)
						crop_df.to_csv(title, mode = 'a', index = True, header=False)
						#print(title)
					total_df = pd.read_csv(title)
					#print(total_df)
					
					# create csv file if first image is blank
					# print(os.path.isfile(title))
					# if os.path.isfile(title) == False:
					# 	print('bing')
					# 	total_df = pd.DataFrame(columns = columns)

					# give titles because when you switch from white to non white column, you get empty dataframe
					# redundant, I know
					total_df.columns = columns
			

					#print(total_df)
					total_df = trimNuclei2(total_df)
					#total_df.columns = columns
					#print(total_df)
					save_dir = wd + '/traces/' + main_image_coord + '_' + stitch_direction[coord_idx] + '_merge.png'
					

					total_i = i#int(stitch_list[coord_idx][0])
					total_j = j#int(stitch_list[coord_idx][2])
					os.remove(stitch_title)
					os.remove(crop_title)
					#graphNucs(total_df, stitch_image, save_dir, total_i, total_j, True )
				temp_image = None

		#sys.exit()

		image1 = None



# total_df = pd.read_csv(combined_title)
# print("Now trimming nuclei...")
# total_df = trimNuclei2(total_df)

# total_df.to_csv(combined_title,index=False, encoding='utf-8-sig')
# print("Done!")


