import numpy as np
import cv2 as cv
from geometry import Point, Points, Line
import pandas as pd

def receive_click_locations(imgpath):

    image = cv.imread(imgpath)
    
    cv.namedWindow("clickfinder")

    locs = []

    def getxy(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONUP:
            colorsBGR = image[y, x]
            # Reversing the OpenCV BGR format to RGB format
            colorsRGB = tuple(reversed(colorsBGR))
            print("x, y, rgb = ({},{}):{} ".format(x, y, colorsRGB))
            cv.drawMarker(image, (x, y), (0, 0, 255), markerType=cv.MARKER_STAR,markerSize=2, thickness=1, line_type=cv.LINE_AA)
            locs.append((x, y))

    cv.setMouseCallback('clickfinder', getxy)

    while (1):
        cv.imshow('clickfinder', image)
        _k = cv.waitKey(10) & 0xFF
        if _k == 27:
            break
        elif _k == 100:
            locs.pop()
            image = cv.imread(imgpath)
            for loc in locs:
                cv.drawMarker(image, loc, (0, 0, 255), markerType=cv.MARKER_STAR,markerSize=2, thickness=1, line_type=cv.LINE_AA)

    cv.destroyAllWindows()
    
    return np.column_stack([np.array(locs), np.zeros(len(locs))])


if __name__ == '__main__':
    scale = "normal_1/S20210601_0016.jpg"
    imgs = ["normal_1/S20210601_0001.jpg",
           # "normal_1/S20210601_0002.jpg",
           # "normal_1/S20210601_0003.jpg",
           # "normal_1/S20210601_0004.jpg",
           # "normal_1/S20210601_0005.jpg",
           # "normal_1/S20210601_0006.jpg",
           # "normal_1/S20210601_0007.jpg",
           # "normal_1/S20210601_0008.jpg",
           # "normal_1/S20210601_0009.jpg",
           # "normal_1/S20210601_0010.jpg",
           # "normal_1/S20210601_0011.jpg",
           # "normal_1/S20210601_0012.jpg",
           # "normal_1/S20210601_0013.jpg",
           # "normal_1/S20210601_0014.jpg",
            "normal_1/S20210601_0015.jpg"]

    print("select two points for scaling")
    ps = receive_click_locations(scale)
    sc = abs(Line(Point(*ps[0]), Point(*ps[1])).vector())

    with open("slotwidths.csv", 'w') as f:
        f.write("filename, width")
    widths = []
    for img in imgs:
        print("select points on top of slot {}".format(1))
        slot_top = Points(receive_click_locations(img))

        print(slot_top)
        print("select points on bottom of slot {}".format(1))
        slot_btm = Points(receive_click_locations(img))
        print(slot_btm)
        top_area = np.trapz(slot_top.y, slot_top.x) / (max(slot_top.x)  - min(slot_top.x))
        btm_area = np.trapz(slot_btm.y, slot_btm.x) / (max(slot_btm.x)  - min(slot_btm.x))
        width = (btm_area - top_area)/sc
        widths.append(width)
        with open("slotwidths.csv", 'a') as f:
            f.write("{}, {}\n".format(img, width))



    df = pd.DataFrame([imgs, widths]).T
    df.columns=["filename", "width"]

    print(df)
    print(df.width.mean())
