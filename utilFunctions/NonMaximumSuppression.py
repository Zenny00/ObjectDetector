from contextlib import suppress
import numpy as np

def non_maximum_suppression(boxes, overlapThreshold):
    #If there are no boxes to operate on return the empty set
    if len(boxes) == 0:
        return []

    #A list that will hold the chosen bboxes
    chosen_boxes = []

    #Grab the coordinates of each of the bboxes and store them in a list
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    bbox_area = (x2 - x1 + 1) * (y2 - y1 + 1)

    #Get the bottom right coordinate of each bbox and use it to rank them
    idxs = np.argsort(y2)
    print(len(idxs))

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        chosen_boxes.append(i)
        suppress = [last]

        for pos in range(0, last):
            j = idxs[pos]

            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            width = max(0, xx2 - xx1 + 1)
            height = max(0, yy2 - yy1 + 1)

            overlap = float(width * height) / bbox_area[j]

            if overlap > overlapThreshold:
                suppress.append(pos)
        
        print("suppress list: ")
        print(suppress)

        idxs = np.delete(idxs, suppress)

    return chosen_boxes