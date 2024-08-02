"""
Loop through png files in a directory and convert to video
The script takes two arguments: the directory containing the images and the output file. It reads all the PNG images in the directory, creates a video writer object, and writes the images to the video file. The video is created with a frame rate of 1 frame per second. You can adjust the frame rate by changing the value `1` in the `cv2.VideoWriter` function.
You can run the script from the terminal as follows:

python3 png2video.py /path/to/images /path/to/output.mp4

Replace `/path/to/images` with the path to the directory containing the PNG images and `/path/to/output.mp4` with the desired output video file path.
Note: Make sure you have OpenCV installed in your Python environment to run the script. You can install it using `pip install opencv-python`.
Note: This function scales depths to 0 and 1 based on min and max of the entire video/image_sequence_directory
"""
import cv2
import os
import sys
import numpy as np

def main():
    if len(sys.argv) < 3:
        print("usage: python3 png2video.py <directory> <output_file>")
        sys.exit(1)
    directory = sys.argv[1]
    output_file = sys.argv[2]
    images = [img for img in os.listdir(directory) if img.endswith(".png")]
    images.sort()           # sort in ascending order
    # set height and width by first frame in directory
    frame = cv2.imread(os.path.join(directory, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
    
    # interate through image list and read images into a list, convert to np.array.
    # scale and convert dtype from uint16 -> uint8
    img_npys = []
    for image in images[0:100]:
        img = cv2.imread(os.path.join(directory, image), cv2.IMREAD_UNCHANGED)
        # print(img, img.shape)
        # video.write(img*8)
        img_npys.append(img)
    img_npys = np.array(img_npys)
    img_npys = (img_npys - img_npys.min()) / (img_npys.max() - img_npys.min()) * 255
    img_npys = img_npys.astype(np.uint8)

    # iterate through image array and reshape to (H, W, C=3) for each image
    # release vid
    for image_np in img_npys:
        video.write(np.moveaxis(np.tile(image_np, (3,1,1)),0, -1))
    video.release()

if __name__ == "__main__":
    main()  # call main function

