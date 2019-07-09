import xml.etree.ElementTree as ET
import glob
import ntpath
import os
import sys
import argparse

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    return interArea / float(boxAArea + boxBArea - interArea)

def metrics(datasetPath, predictionPath, threshold=0.5):
    if os.path.isdir(datasetPath) == False:
        sys.exit('Dataset path is not valid')
    if os.path.isdir(predictionPath) == False:
        sys.exit('Prediction path is not valid')

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    groundtruthLentgh = 0
    predictionLentgh = 0

    for file in glob.glob("{}/*.xml".format(datasetPath)):
        try:
            xmltree = ET.parse(file)
            objects = xmltree.findall("object")
            dBoxes = []
            for object_iter in objects:
                bndbox = object_iter.find("bndbox")
                dBoxes.append([int(it.text) for it in bndbox])

            xmltree = ET.parse('{}/'.format(predictionPath) + ntpath.basename(file))
            objects = xmltree.findall("object")
            pBoxes = []
            for object_iter in objects:
                bndbox = object_iter.find("bndbox")
                pBoxes.append([int(it.text) for it in bndbox])
        except:
            print('XML file {} not found or corrupted '.format(ntpath.basename(file)))

        groundtruthLentgh += len(dBoxes)
        predictionLentgh += len(pBoxes)

        if len(pBoxes) == 0 and len(dBoxes):
            TN += 1

        for i in range(len(dBoxes)):
            for j in range(len(pBoxes)):
                iouScore = iou(dBoxes[i], pBoxes[j])
                if iouScore > threshold:
                    TP += 1

    FN = groundtruthLentgh - TP
    FP = predictionLentgh - TP

    return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN, 'TPR': TP/(TP+FN), 'FPR': FP/(FP+TN), 'accuracy': ((TP+TN)/(TP+TN+FN+FP)) }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, help="path to directory with predictions")
    parser.add_argument("-g", type=str, help="path to directory with ground truth")
    args = parser.parse_args()

    result = metrics(args.g,args.d)
    for key in result:
        print("{}: {}".format(key, result[key]))