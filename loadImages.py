"""
load data directory

v. 2021 02 21

"""

# define data directory and image file range
image_directory = '/home/allen/projects/training-data/data/images'

#############################################################################
#############################################################################
imageDF = pa.loadDir(image_directory)
print("loaded {} images from {}".format(len(imageDF),image_directory))
