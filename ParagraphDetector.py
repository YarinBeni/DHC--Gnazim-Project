import os

import numpy as np
import cv2
import pytesseract
import re

# Based on https://stackoverflow.com/questions/57249273/how-to-detect-paragraphs-in-a-text-document-image-for-a-non-consistent-text-stru


# def rlsa(image, threshold):
#     # Apply horizontal RLSA
#     horizontal_rlsa = np.copy(image)
#     for row in range(image.shape[0]):
#         run_start = 0
#         for col in range(image.shape[1]):
#             if image[row, col] == 255:  # White pixel
#                 if col - run_start <= threshold:
#                     horizontal_rlsa[row, run_start:col] = 255
#                 run_start = col
#
#     # Apply vertical RLSA
#     vertical_rlsa = np.copy(horizontal_rlsa)
#     for col in range(horizontal_rlsa.shape[1]):
#         run_start = 0
#         for row in range(horizontal_rlsa.shape[0]):
#             if horizontal_rlsa[row, col] == 255:  # White pixel
#                 if row - run_start <= threshold:
#                     vertical_rlsa[run_start:row, col] = 255
#                 run_start = row
#
#     return vertical_rlsa


import cv2
import re
import pytesseract


class ImageProcessor:
    def __init__(self, image, lang='heb'):
        self.image = image
        self.lang = lang
        self.resize_and_plot(self.image, "original_image")

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
        self.ocr_of_all_text_no_prero = re.sub(r'\n+', ' ', pytesseract.image_to_string(self.image, lang=self.lang))
        # print(f"\nocr of all text with NO preprocess: {self.ocr_of_all_text_no_prero}\n")

        thresh = self.pre_contour_preprocess()
        contours, dilate, rectangles = self.roi_detection(thresh)
        thresh = self.post_contour_preprocess(dilate)
        self.ocr_of_all_text_prepro = re.sub(r'\n+', ' ', pytesseract.image_to_string(thresh, lang=self.lang))
        # print(f"ocr of all text with preprocess: {self.ocr_of_all_text_prepro}\n")
        return self.create_croped_roi(contours, thresh)

    @staticmethod
    def resize_and_plot(image, name="", show_image=False):
        screen_res = 1920, 1080
        scale_width = screen_res[0] / image.shape[1]
        scale_height = screen_res[1] / image.shape[0]
        scale = min(scale_width, scale_height) * 0.5
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        if show_image:
            cv2.imshow('Resized Image', resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # if name:
        #     # Construct the full path where the image will be saved
        #     file_path = rf"C:\Users\yarin\PycharmProjects\DHC\GnazimProject\outputs\{name}.png"
        #
        #     # Save the image
        #     cv2.imwrite(file_path, resized)

    def create_croped_roi(self, contours, image):
        copy_image = image.copy()
        cropped_images = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 15 and h > 40:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cropped = copy_image[y:y + h, x:x + w]
                cropped_images.append(
                    (cropped, (x, y, x + w, y + h), pytesseract.image_to_string(cropped, lang=self.lang)))
                # cropedimage, top left and bottom right corner of image in original images,ocr text
        self.resize_and_plot(image, "rectangles_detected_image")
        self.rectangles_detected_image = image
        return cropped_images[::-1]

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
        self.resize_and_plot(dilate, "first_iter_of_detection_algo")

        while flag:
            # self.resize_and_plot(dilate)
            iter_param += 2
            dilate = cv2.dilate(thresh, kernel, iterations=iter_param)
            contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectangles = [cv2.boundingRect(c) for c in contours]
            flag = self.exists_two_close_rectangles(rectangles)
        self.resize_and_plot(dilate, "last_iter_of_detection_algo")

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
        # Initializing the dictionary with default values
        paragraphs_detection_process_completed = True
        extracted_ocr_data = {
            "ocr_writen_on": "",
            "ocr_writen_by": "",
            "ocr_main_content": "",
            "ocr_additional_content": "",
            "ocr_writen_on_coords": "",
            "paragraphs_detection_successes": paragraphs_detection_process_completed
        }
        image_width = self.image.shape[1]  # Image width is the second value of shape
        middle_x_point = image_width / 2

        # Create ocr_list by filtering out items where ocr text was found in the bounding box(len(item[2]) < 1 )
        ocr_list = list(filter(lambda item: len(item[2]) > 0, cropped_images))

        # If written_on_candidates list is not empty, process it
        if ocr_list:
            # Create written_on_candidates list
            if len(ocr_list)>1:
                written_on_candidates = sorted(
                    filter(lambda item: item[1][0] > middle_x_point, ocr_list),
                    key=lambda item: (item[1][1], -item[1][0])
                )
            else:
                written_on_candidates = ocr_list
            if written_on_candidates:
                written_on_image, written_on_coords, written_on_text = written_on_candidates[0]
                written_on_text = [line for line in written_on_text.split("\n") if line.strip()]
                extracted_ocr_data["ocr_writen_on"] = written_on_text[0]
                extracted_ocr_data["ocr_writen_on_coords"] =  re.sub(r'[()]', '', str(written_on_coords))
                self.resize_and_plot(written_on_image, "written_on_image")
                if len(written_on_text) > 1:
                    extracted_ocr_data["ocr_writen_by"] = written_on_text[1]
            else:
                paragraphs_detection_process_completed = False
            # Create main_content_candidates list
            main_content_candidates = sorted(ocr_list, key=lambda x: len(x[2]), reverse=True)
            if main_content_candidates:
                # Extracting main content from main_content_candidates list
                main_content_text = main_content_candidates[0][2]
                extracted_ocr_data["ocr_main_content"] = main_content_text.replace("\n", "")
                self.resize_and_plot(main_content_candidates[0][0], "ocr_main_content_image")
                # Check if the next longest text is different from 'written_on' text
                if len(main_content_candidates)>1:
                    additional_content_text = main_content_candidates[1][2] if main_content_candidates[1][2] != \
                                                                               extracted_ocr_data[
                                                                                   "ocr_writen_on"] + extracted_ocr_data[
                                                                                   "ocr_writen_by"] else ""
                    extracted_ocr_data["ocr_additional_content"] = additional_content_text
                    self.resize_and_plot(main_content_candidates[1][0], "ocr_additional_content_image")
            else:
                paragraphs_detection_process_completed = False
        else:
            self.resize_and_plot(self.rectangles_detected_image, "") # SHOW IMAGE THAT DETECTION FAILED
            paragraphs_detection_process_completed = False
        extracted_ocr_data["paragraphs_detection_successes"] = paragraphs_detection_process_completed
        return extracted_ocr_data

    def run(self, log_metadata=False):
        roi_list = self.get_paragraph_boundingbox()
        result = self.plot_roi_and_ocr(roi_list)
        result["ocr_all_text_preprocess"] = self.ocr_of_all_text_prepro
        result["ocr_all_text_no_preprocess"] = self.ocr_of_all_text_no_prero
        if result['paragraphs_detection_successes'] and len(result["ocr_writen_by"]) + len(result["ocr_writen_on"]) > 1:
            cut_index = len(result["ocr_writen_by"]) + len(result["ocr_writen_on"]) + len(
                result["ocr_additional_content"])
            result["ocr_main_content_all_text_preprocess"] = self.ocr_of_all_text_prepro[cut_index:]
            result["ocr_main_content_all_text_no_preprocess"] = self.ocr_of_all_text_no_prero[cut_index:]
        else:
            result["ocr_main_content_all_text_preprocess"] = ""
            result["ocr_main_content_all_text_no_preprocess"] = ""

        if log_metadata:
            print("-" * 110)
            print(
                "--------OCR extracted meta data from image is:----------------------------------------------------------------")
            for k in result.keys():
                print(k, ": ", result[k])
            print("-" * 110)
            print()
        return result

# Use it like this:
# path= 'POC_sample3_handwriten.tif'  # Hand Writen Example ALSO WORKS! GOOD
# path ='POC_sample2_withnoise.tif'  # GOOD
# path =  'SAMPLE_Long_Author04.tif'# GOOD
# path =    'POC_sample5_withnoise_Author.tif'  # GOOD FIXED PROBLEM OVERLAP RECTANGLES but get all rectangles
#
# img=cv2.imread(path, 0)
# processor = ImageProcessor(img)
# processor.run(True)

# https://pypi.org/project/HspellPy/ Maybe to use it as spellchecker

# todo: to be flexible with permutation חיים גורי and גורי חיים , Need list of stages from yael
#  open refine api, clustering algorithm editing distance:
#  https://openrefine.org/docs/technical-reference/clustering-in-depth
# todo: confusion matrix for mistake in classify letters per image

import re
