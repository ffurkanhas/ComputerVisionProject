import os
from fnmatch import fnmatch
import scipy.io
from PIL import Image
from tqdm import tqdm

noneCount = 0
imagePaths = "/run/media/toorn/New Volume/CarDataSet/CompCars/data/data/image/"
rootPath = '/run/media/toorn/New Volume/SonData/'
trainPath = rootPath + "train"
validPath = rootPath + "valid"
testPath = rootPath + "test"
size = 100, 100


def walkdir(folder):
    for dirpath, dirs, files in os.walk(folder):
        for filename in files:
            yield os.path.abspath(os.path.join(dirpath, filename))


def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}


def copyImages():
    pattern = "*.jpg"
    carTypeDict, typesCount = getVehicleTypeDictionary()
    totalCarCount = 0
    print(typesCount)
    for i in typesCount:
        totalCarCount += typesCount[i]

    print("Total car count: ", totalCarCount)
    if not os.path.exists(trainPath):
        os.makedirs(trainPath)

    if not os.path.exists(validPath):
        os.makedirs(validPath)

    if not os.path.exists(testPath):
        os.makedirs(testPath)

    withoutNoneTypeCount = 0
    typesCount = without_keys(typesCount, {'none'})
    for i in typesCount:
        withoutNoneTypeCount += typesCount[i]
    print("Without none types: ", withoutNoneTypeCount)

    filecounter = 0
    with tqdm(walkdir(imagePaths), unit=" files") as pbar:
        for path, subdirs, files in os.walk(imagePaths):
            for name in files:
                filecounter += 1
                pbar.update(1)

    pbar = tqdm(total=filecounter, unit="files")

    tempTypeCount = typesCount
    for i in tempTypeCount:
        tempTypeCount[i] = 0

    testCount = 0

    for path, subdirs, files in os.walk(imagePaths):
        for name in files:
            if fnmatch(name, pattern):
                modelId = path.split('/').__getitem__(11)
                carType = carTypeDict[modelId]
                flag = False
                if carType != 'none':
                    tempTypeCount[carType] += 1
                    if(tempTypeCount[carType] < 1200):
                        flag = True
                        targetFolderPath = trainPath + "/" + carTypeDict[modelId] + "/"
                        new_name = carTypeDict[modelId] + "_" + str(tempTypeCount[carType]) + ".jpg"

                else:
                    flag = True
                    testCount += 1
                    targetFolderPath = testPath + "/"
                    new_name = "test_" + str(testCount) + ".jpg"

                if flag:
                    if not os.path.exists(targetFolderPath):
                        os.makedirs(targetFolderPath)

                    img = Image.open(os.path.join(path, name))

                    newImagePath = targetFolderPath + new_name
                    img.save(newImagePath, "JPEG", optimize=True)

                pbar.set_postfix(iteratedFile=os.path.join(path, name), refresh=True)
                pbar.update(1)
    print("Total: ", tempTypeCount)


def readImages():
    pattern = "*.jpg"
    vehicleList = list()
    carTypeDict = getVehicleTypeDictionary()

    for path, subdirs, files in os.walk(imagePaths):
        for name in files:
            if fnmatch(name, pattern):
                brandId = path.split('/').__getitem__(10)
                modelId = path.split('/').__getitem__(11)
                tempDictionary = dict()
                tempDictionary['brandId'] = brandId
                tempDictionary['modelId'] = modelId
                tempDictionary['imagePath'] = os.path.join(path, name)
                tempDictionary['carType'] = carTypeDict[modelId]
                vehicleList.append(tempDictionary)
    return vehicleList


def getTypesFromMatFile():
    carTypeFile = '/run/media/toorn/New Volume/CarDataSet/CompCars/data/data/misc/car_type.mat'

    mat = scipy.io.loadmat(carTypeFile)

    typesMatArray = mat.get('types')
    types = dict()
    types[0] = 'none'

    for i in range(len(typesMatArray[0])):
        types[i+1] = str(typesMatArray[0][i]).replace('[', '').replace(']', '').replace('\'', '')
    return types


def getVehicleTypeDictionary():
    types = getTypesFromMatFile()
    typesCount = dict()
    carTypeDictionary = dict()
    carTypeListFilePath = '/run/media/toorn/New Volume/CarDataSet/CompCars/data/data/misc/attributes.txt'

    for carType in types:
        typesCount[types[carType]] = 0

    with open(carTypeListFilePath) as f:
        lines = f.readlines()

    for i in range(1, len(lines)):
        line = lines[i].rstrip()
        line = line.split(' ')
        carModel = line[0]
        carType = line[5]
        carTypeDictionary[carModel] = types.__getitem__(int(carType))
        typesCount[types.__getitem__(int(carType))] += 1

    return carTypeDictionary, typesCount


copyImages()