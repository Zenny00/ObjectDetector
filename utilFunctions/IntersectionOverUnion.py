def compute_intersection_over_union(bboxA, bboxB):
    #Get the four corners of the intersection
    Ax = max(bboxA[0], bboxB[0])
    Ay = max(bboxA[1], bboxB[1])
    Bx = max(bboxA[2], bboxB[2])
    By = max(bboxA[3], bboxB[3])

    area_intersection = max(0, (Bx - Ax + 1)) * max(0, (By - Ay + 1))
    #area = l*h, get the area of each bounding box
    area_bboxA = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    area_bboxB = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)

    #Calculate the intersection over union by dividing the area of the intersection
    #by the area of the union (Notice the area of the intersection is subtracted from
    # the union to avoid double counting)
    intersection_over_union = (area_intersection / float(area_bboxA + area_bboxB - area_intersection))

    return intersection_over_union