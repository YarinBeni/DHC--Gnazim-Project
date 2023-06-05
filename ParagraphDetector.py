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




# Load image, grayscale, Gaussian blur, Otsu's threshold
#image = cv2.imread('POC_sample3_handwriten.tif', 0) # Hand Writen Example
image = cv2.imread('POC_sample2_withnoise.tif', 0) #  Digital Typed Example
#image=cv2.imread('SAMPLE_Long_Author04.tif',0)
#image = cv2.imread('POC_sample5_withnoise_Author.tif', 0) # Digital Typed Example

def get_paragraph_boundingbox():
    global cropped_images, cropped
    blur = cv2.medianBlur(image, 5)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh = cv2.medianBlur(thresh, 5)
    thresh = cv2.medianBlur(thresh, 5)
    thresh = cv2.medianBlur(thresh, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=10)
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cropped_images = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w > 15 and h > 40):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = image[y:y + h, x:x + w]
            cropped_images.append(cropped)
    cv2.imshow('thresh', thresh)
    cv2.imshow('dilate', dilate)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cropped_images = sorted(cropped_images, key=lambda img: (img.shape[0], -img.shape[1]))
    return cropped_images


roi_list=get_paragraph_boundingbox(image)


def plot_roi_and_ocr(cropped_images):
    # Plot the cropped images
    for i, cropped in enumerate(cropped_images):
        ocr_text = re.sub(r'\n+', ' ', pytesseract.image_to_string(cropped, lang='heb'))
        print(f"i: {i},text: {ocr_text}\n")
        cv2.imshow(f'Cropped {i}', cropped)

        if len(ocr_text) > 1:
            print(f"i: {i},text: {ocr_text}\n")
            # cv2.imshow(f'Cropped {i}', cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


plot_roi_and_ocr(cropped_images)