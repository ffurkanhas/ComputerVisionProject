from sklearn.neighbors import KNeighborsClassifier
from skimage import feature
import imutils
import cv2
from skimage import data, exposure
from random import shuffle
import numpy as np

import preprocess as pre

# image = cv2.imread(readImages().__getitem__(0))
# image2 = cv2.imread(readImages().__getitem__(1))
#
# fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
#
# ax1.axis('off')
# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.set_title('Input image')
#
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#
# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# plt.show()


vehicleList = pre.readImages()
shuffle(vehicleList)


def trainData():
    print('training Started...')
    data = []
    labels = []
    count = 0
    dataFile = open('data.txt', 'w')
    labelFile = open('labels.txt', 'w')

    for car in vehicleList[:40000]:
        # extract the make of the car
        make = car.get('carType')
        imagePath = car.get('imagePath')

        # load the image, convert it to grayscale, and detect edges
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = imutils.auto_canny(gray)

        # find contours in the edge map, keeping only the largest one which
        # is presmumed to be the car logo
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2(cnts) else cnts[1]
        c = max(cnts, key=cv2.contourArea)

        # extract the logo of the car and resize it to a canonical width
        # and height
        (x, y, w, h) = cv2.boundingRect(c)
        logo = gray[y:y + h, x:x + w]
        logo = cv2.resize(logo, (200, 100))

        # extract Histogram of Oriented Gradients from the logo
        H = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
                        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")

        # update the data and labels
        data.append(H)
        dataFile.write(str(H))
        dataFile.write("\n")
        labels.append(make)
        labelFile.write(str(make))
        labelFile.write("\n")

        count += 1
    return data, labels


# with open('YedekData/data.txt', 'r') as file:
#   filedata = file.read()
#
# # Replace the target string
# filedata = filedata.replace('\ufeff', '')
#
# # Write the file out again
# with open('YedekData/data2.txt', 'w') as file:
#   file.write(filedata)
#
# import pandas
# df = pandas.read_table('YedekData/data2.txt',
#                        delim_whitespace=True,
#                        header=None,
#                        engine='python')
# dataTemp = np.array(df)
# labelsTemp = np.genfromtxt('YedekData/labels2.txt', dtype=str)

data, labels = trainData()

print("[INFO] training classifier...")
model = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
model.fit(data, labels)
print("[INFO] evaluating...")

pathList = list()
for car in vehicleList[500:600]:
    pathList.append(car.get('imagePath'))


for (i, imagePath) in enumerate(pathList):
    # load the test image, convert it to grayscale, and resize it to
    # the canonical size
    if i < 10:
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logo = cv2.resize(gray, (200, 100))

        # extract Histogram of Oriented Gradients from the test image and
        # predict the make of the car
        (H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
                                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualise=True)
        pred = model.predict(H.reshape(1, -1))[0]

        # visualize the HOG image
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")
        cv2.imshow("HOG Image #{}".format(i + 1), hogImage)

        # draw the prediction on the test image and display it
        cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 3)
        imageNewName = str(i) + '_' + str(pred.title()) + '_knn.png'
        cv2.imwrite(imageNewName, image)
        hogImageNewName = str(i) + '_hog_knn.png'
        cv2.imwrite(hogImageNewName, hogImage)
