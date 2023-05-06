from PIL import Image
import os
import numpy as np


# Set the path to the directory containing the images
path_to_directory = './task3/dataset/food'

# Initialize empty lists to store the statistics and sizes
mean_red = []
mean_green = []
mean_blue = []
var_red = []
var_green = []
var_blue = []
widths = []
lengths = []

count = 0
# Loop through each image in the directory
for filename in os.listdir(path_to_directory):
    count += 1
    print(count)
    # Load the image
    img = Image.open(os.path.join(path_to_directory, filename))

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Compute the mean of each channel
    mean_red.append(np.mean(img_array[:,:,0]))
    mean_green.append(np.mean(img_array[:,:,1]))
    mean_blue.append(np.mean(img_array[:,:,2]))

    # Compute the variance of each channel
    var_red.append(np.var(img_array[:,:,0]))
    var_green.append(np.var(img_array[:,:,1]))
    var_blue.append(np.var(img_array[:,:,2]))

    # Get the width and length of the image
    width, length = img.size
    widths.append(width)
    lengths.append(length)

# Compute the overall statistics across all images
overall_mean_red = np.mean(mean_red)
overall_mean_green = np.mean(mean_green)
overall_mean_blue = np.mean(mean_blue)
overall_var_red = np.mean(var_red)
overall_var_green = np.mean(var_green)
overall_var_blue = np.mean(var_blue)
overall_std_red = np.sqrt(overall_var_red)
overall_std_green = np.sqrt(overall_var_green)
overall_std_blue = np.sqrt(overall_var_blue)
min_width = np.min(widths)
max_width = np.max(widths)
min_length = np.min(lengths)
max_length = np.max(lengths)

print('Overall mean pixel value of the red channel: ', overall_mean_red)
print('Overall mean pixel value of the green channel: ', overall_mean_green)
print('Overall mean pixel value of the blue channel: ', overall_mean_blue)
print('Overall variance of the red channel: ', overall_var_red)
print('Overall variance of the green channel: ', overall_var_green)
print('Overall variance of the blue channel: ', overall_var_blue)
print('Overall standard deviation of the red channel: ', overall_std_red)
print('Overall standard deviation of the green channel: ', overall_std_green)
print('Overall standard deviation of the blue channel: ', overall_std_blue)
print('Minimum width: ', min_width)
print('Maximum width: ', max_width)
print('Minimum length: ', min_length)
print('Maximum length: ', max_length)