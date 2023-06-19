import numpy as np
import cv2
import pytesseract
import re


# Based on https://stackoverflow.com/questions/57249273/how-to-detect-paragraphs-in-a-text-document-image-for-a-non-consistent-text-stru


def rlsa(image, threshold):
    # Apply horizontal RLSA
    horizontal_rlsa = np.copy(image)
    for row in range(image.shape[0]):
        run_start = 0
        for col in range(image.shape[1]):
            if image[row, col] == 255:  # White pixel
                if col - run_start <= threshold:
                    horizontal_rlsa[row, run_start:col] = 255
                run_start = col

    # Apply vertical RLSA
    vertical_rlsa = np.copy(horizontal_rlsa)
    for col in range(horizontal_rlsa.shape[1]):
        run_start = 0
        for row in range(horizontal_rlsa.shape[0]):
            if horizontal_rlsa[row, col] == 255:  # White pixel
                if row - run_start <= threshold:
                    vertical_rlsa[run_start:row, col] = 255
                run_start = row

    return vertical_rlsa


import cv2
import re
import pytesseract


class ImageProcessor:
    def __init__(self, image_path, lang='heb'):
        self.image = cv2.imread(image_path, 0)
        self.lang = lang

    @staticmethod
    def minimum_abs_difference(rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        differences = [abs(x1 - x2), abs((x1 + w1) - x2), abs(x1 - (x2 + w2)), abs((x1 + w1) - (x2 + w2))]
        return min(differences)

    def exists_two_close_rectangles(self, rectangles):
        height, width = self.image.shape
        height_epsilon, width_epsilon = height * 0.002, width * 0.002
        for i, rect1 in enumerate(rectangles):
            for rect2 in rectangles[i + 1:]:
                min_distance = self.minimum_abs_difference(rect1, rect2)
                if min_distance < height_epsilon or min_distance < width_epsilon:
                    return True
        return False

    def get_paragraph_boundingbox(self):
        ocr_of_all_text = re.sub(r'\n+', ' ', pytesseract.image_to_string(self.image, lang=self.lang))
        print(f"\nocr of all text with NO preprocess: {ocr_of_all_text}\n")

        thresh = self.pre_contour_preprocess()
        contours, dilate, rectangles = self.roi_detection(thresh)
        thresh = self.post_contour_preprocess(dilate)

        ocr_of_all_text = re.sub(r'\n+', ' ', pytesseract.image_to_string(thresh, lang=self.lang))
        print(f"\nocr of all text with preprocess: {ocr_of_all_text}\n")

        cropped_images = self.create_croped_roi(contours, thresh)
        cropped_images = sorted(cropped_images, key=lambda img: (img.shape[0], -img.shape[1]))
        return cropped_images

    @staticmethod
    def resize_and_plot(image):
        screen_res = 1920, 1080
        scale_width = screen_res[0] / image.shape[1]
        scale_height = screen_res[1] / image.shape[0]
        scale = min(scale_width, scale_height) * 0.5
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow('Resized Image', resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def create_croped_roi(self, contours, image):
        copy_image = image.copy()
        cropped_images = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 15 and h > 40:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cropped = copy_image[y:y + h, x:x + w]
                cropped_images.append(cropped)
        self.resize_and_plot(image)

        return cropped_images

    def post_contour_preprocess(self, dilate):
        test = (-self.image + dilate)
        test = cv2.medianBlur(test, 5)
        test = cv2.medianBlur(test, 5)
        test = cv2.threshold(test, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return test

    def roi_detection(self, thresh):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        iter_param = 4
        dilate = cv2.dilate(thresh, kernel, iterations=iter_param)
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = [cv2.boundingRect(c) for c in contours]
        flag = self.exists_two_close_rectangles(rectangles)
        self.resize_and_plot(dilate)

        while flag:
            self.resize_and_plot(dilate)
            iter_param += 2
            dilate = cv2.dilate(thresh, kernel, iterations=iter_param)
            contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectangles = [cv2.boundingRect(c) for c in contours]
            flag = self.exists_two_close_rectangles(rectangles)
        return contours, dilate, rectangles

    def pre_contour_preprocess(self):
        blur = cv2.medianBlur(self.image, 5)
        blur = cv2.GaussianBlur(blur, (5, 5), 0)
        thresh = cv2.medianBlur(blur, 5)
        thresh = cv2.medianBlur(thresh, 5)
        thresh = cv2.medianBlur(thresh, 5)
        thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return thresh

    def plot_roi_and_ocr(self, cropped_images):
        for i, cropped in enumerate(cropped_images):
            ocr_text = re.sub(r'\n+', ' ', pytesseract.image_to_string(cropped, lang=self.lang))
            if len(ocr_text) > 1:
                print(f"i: {i},text: {ocr_text}\n")
            self.resize_and_plot(cropped)

    def run(self):
        roi_list = self.get_paragraph_boundingbox()
        self.plot_roi_and_ocr(roi_list)


# Use it like this:
# path= 'POC_sample3_handwriten.tif'  # Hand Writen Example ALSO WORKS! GOOD
path ='POC_sample2_withnoise.tif'  # GOOD
# path =  'SAMPLE_Long_Author04.tif'# GOOD
# path =    'POC_sample5_withnoise_Author.tif'  # PROBLEM OVERLAP RECTANGLES but get all rectangles
#
processor = ImageProcessor(path)
processor.run()

