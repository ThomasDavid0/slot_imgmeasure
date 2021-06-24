import numpy as np
import cv2 as cv
from geometry import Point, Points, Line
import pandas as pd
import copy


def receive_click_locations(windname, img, colour):

    image = copy.deepcopy(img)

    locs = []

    def getxy(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONUP:
            colorsBGR = image[y, x]
            # Reversing the OpenCV BGR format to RGB format
            colorsRGB = tuple(reversed(colorsBGR))
            print("x, y, rgb = ({},{}):{} ".format(x, y, colorsRGB))
            cv.drawMarker(image, (x, y), colour, markerType=cv.MARKER_STAR,
                          markerSize=2, thickness=1, line_type=cv.LINE_AA)
            locs.append((x, y))

    cv.setMouseCallback(windname, getxy)

    while True:
        cv.imshow(windname, image)
        _k = cv.waitKey(10) & 0xFF
        if _k == 27:
            break
        elif _k == 100:
            locs.pop()
            image = copy.deepcopy(img)
            for loc in locs:
                cv.drawMarker(image, loc, colour, markerType=cv.MARKER_STAR,
                              markerSize=2, thickness=1, line_type=cv.LINE_AA)

    return np.column_stack([np.array(locs), np.zeros(len(locs))]), image


def measure_slot(windname, imgpath):

    image = cv.imread(imgpath)

    print("select top points")
    slot_top, top_img = receive_click_locations(windname, image, (255, 0, 0))
    print("select btm points")
    slot_btm, top_btm_img = receive_click_locations(
        windname, top_img, (0, 0, 255))
    slot_top = Points(slot_top)
    slot_btm = Points(slot_btm)
#    cv.imwrite("output/" + imgpath.replace(".jpg", "_topbtm.jpg"), top_btm_img)

    top_area = np.trapz(slot_top.y, slot_top.x) / (max(slot_top.x) - min(slot_top.x))
    btm_area = np.trapz(slot_btm.y, slot_btm.x) / (max(slot_btm.x) - min(slot_btm.x))
    width = (btm_area - top_area)
    return width, top_btm_img




def measure_slot_images(windname, scale, imgs, label):
    with open("output/" + label + "/slotwidths.csv", 'w') as f:
        f.write("filename, width\n")


    print("select two points for scaling")
    ps, scale_image = receive_click_locations(
        windname, cv.imread(scale), (0, 255, 0))
    cv.imwrite("output/normal_1/scale.jpg", scale_image)

    sc = abs(Line(Point(*ps[0]), Point(*ps[1])).vector())

    widths = []
    for img in imgs:
        width, slot_point_image = measure_slot(windname, img)
        width = width / sc
        print(img, str(width))
        widths.append(width)
        cv.imwrite("output/" + img.replace(".jpg",
                   "_topbtm.jpg"), slot_point_image)
        with open("output/" + label + "/slotwidths.csv", 'a') as f:
            f.write("{}, {}\n".format(img, width))

    df = pd.DataFrame([imgs, widths]).T
    df.columns = ["filename", "width"]

    print(df)
    print(df.width.mean())



if __name__ == '__main__':
    normal_scale = "normal_1/S20210601_0016.jpg"
    normal_imgs = ["normal_1/S20210601_0001.jpg",
            "normal_1/S20210601_0002.jpg",
            "normal_1/S20210601_0003.jpg",
            "normal_1/S20210601_0004.jpg",
            "normal_1/S20210601_0005.jpg",
            "normal_1/S20210601_0006.jpg",
            "normal_1/S20210601_0007.jpg",
            "normal_1/S20210601_0008.jpg",
            "normal_1/S20210601_0009.jpg",
            "normal_1/S20210601_0010.jpg",
            "normal_1/S20210601_0011.jpg",
            "normal_1/S20210601_0012.jpg",
            "normal_1/S20210601_0013.jpg",
            "normal_1/S20210601_0014.jpg",
            "normal_1/S20210601_0015.jpg"
            ]
    tangential_scale = "tangential_3/S20210601_0008.jpg"
    tangential_imgs=[
        "tangential_3/S20210601_0001.jpg",
        "tangential_3/S20210601_0002.jpg",
        "tangential_3/S20210601_0003.jpg",
        "tangential_3/S20210601_0004.jpg",
        "tangential_3/S20210601_0005.jpg",
        "tangential_3/S20210601_0006.jpg",
        "tangential_3/S20210601_0007.jpg",
    ]
    windname = "clickfinder"
    cv.namedWindow(windname, cv.WINDOW_NORMAL)
    cv.resizeWindow(windname, 1200, 1200)
    measure_slot_images(windname, normal_scale, normal_imgs, "normal_1")
    #measure_slot_images(windname, tangential_scale, tangential_imgs, "tangential_3")

    cv.destroyAllWindows()
