"""
load data directory

v. 2021 02 21

"""

# define data directory and image file range
image_directory = '/home/allen/projects/data/Recorder-Stream-1829.tif_Files'

#############################################################################
#############################################################################
imageDF = pa.loadDir(image_directory)
print("loaded {} images from {}".format(len(imageDF),image_directory))
