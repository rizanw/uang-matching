
import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import argparse


def uang_matching():
    # load template
    template_data = []
    template_files = glob.glob('template/*.jpg', recursive=True)
    print("template loaded:", template_files)
    # prepare template
    for template_file in template_files:
        tmp = cv2.imread(template_file)
        tmp = imutils.resize(tmp, width=int(tmp.shape[1]*0.5))  # scalling
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)  # grayscale
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        tmp = cv2.filter2D(tmp, -1, kernel) #sharpening
        tmp = cv2.blur(tmp, (3, 3))  # smoothing
        tmp = cv2.Canny(tmp, 50, 200)  # Edge with Canny
        nominal = template_file.replace('template\\', '').replace('.jpg', '')
        template_data.append({"glob":tmp, "nominal":nominal})

    # template matching
    for image_glob in glob.glob('test/*.jpg'):
        for template in template_data:
            image_test = cv2.imread(image_glob)
            (tmp_height, tmp_width) = template['glob'].shape[:2]
            cv2.imshow("Template", template['glob'])  

            image_test_p = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY) 

            image_test_p = cv2.Canny(image_test_p, 50, 200)
            cv2.imshow("Step: edge with canny", image_test_p)

            found = None
            thershold = 0.4
            for scale in np.linspace(0.2, 1.0, 20)[::-1]: 
                # scalling uang
                resized = imutils.resize(
                    image_test_p, width=int(image_test_p.shape[1] * scale))
                r = image_test_p.shape[1] / float(resized.shape[1]) 
                if resized.shape[0] < tmp_height or resized.shape[1] < tmp_width:
                    break

                # template matching
                result = cv2.matchTemplate(resized, template['glob'], cv2.TM_CCOEFF_NORMED)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, r)
                    if maxVal >= thershold: 
                        print("money:", template['nominal'], "detected")
            if found is not None: 
                (maxVal, maxLoc, r) = found
                (startX, startY) = (int(maxLoc[0]*r), int(maxLoc[1] * r))
                (endX, endY) = (
                    int((maxLoc[0] + tmp_width) * r), int((maxLoc[1] + tmp_height) * r))
                if maxVal >= thershold:
                    cv2.rectangle(image_test, (startX, startY),
                                  (endX, endY), (0, 0, 255), 2)
                cv2.imshow("Result", image_test)
            cv2.waitKey(0)


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("-s", "--single-img", type=bool, const=True, nargs='?')
    args = vars(arg.parse_args())
    # print(args)
    uang_matching()
