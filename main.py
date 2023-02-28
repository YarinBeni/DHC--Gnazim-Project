import csv
import os
import pytesseract
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# this is the image processing we need to do an image before we use tesartct to optimze it
# image = cv2.imread('POC_sample2_withnoise.tif', 0)
# blurred = cv2.GaussianBlur(image, (5, 5), 0)
# _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# attempt to add cluster to the images to find handwritten-machine typed
# import os
# from sklearn.cluster import MiniBatchKMeans
#
# folder_name = "CD00085"
# image_data = []
# image_names = []
# batch_size = 10
#
# kmeans = MiniBatchKMeans(n_clusters=2,batch_size=batch_size)
# for file in os.listdir(folder_name):
#     if file.endswith(".tif"):
#         print(os.path.join(folder_name, file))
#         image = cv2.imread(os.path.join(folder_name, file))
#         image_data.append(image.reshape(-1))
#         image_names.append(file)
#         if len(image_data) == batch_size and len(image_data) != 0:
#             kmeans.partial_fit(image_data)
#             image_data = []
#
# clusters = kmeans.predict(image_data)
# with open("clusters.csv", "w") as f:
#     writer = csv.writer(f)
#     for i in range(len(image_names)):
#         writer.writerow([image_names[i], clusters[i]])
#
# print(clusters)


def plot_preprocess_images(image, blurred, thresholded):
    samples = [{"image": image, "Title": "Original Image", "Histogram Title": "Original Histogram"},
               {"image": blurred, "Title": "Blurred Image", "Histogram Title": "Blurred Histogram"},
               {"image": thresholded, "Title": "Thresholded Image", "Histogram Title": "Thresholded Histogram"}]
    for i in range(len(samples)):
        dict = samples[i]
        plt.subplot(3, 2, 1 + i * 2), plt.imshow(dict["image"], 'gray')
        plt.title(dict["Title"]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 2, 2 + i * 2), plt.hist(dict["image"].ravel(), bins=256, range=(0, 255))
        plt.title(dict["Histogram Title"]), plt.xticks([]), plt.yticks([])
        im = dict["image"]
        print("min value of the image:", np.min(im))
        # Count the number of pixels with a value greater than 254
        print("Number of pixels with a value greater than 254:", np.sum(im > 254))
        print("Number of pixels with a value in range: 4-254:", np.sum((im > 3) & (im <= 254)))
        print("Number of pixels with a value less than 4:", np.sum(im < 4))
    plt.show()


# plot_preprocess_images(image,blurred,thresholded)

def read_txt_to_csv(im, filename):
    global writer, line
    text = pytesseract.image_to_string(im, lang='heb')

    print(text, "\n")
    with open(filename + 'output.csv', 'w', newline='') as th_csvfile:
        writer = csv.writer(th_csvfile)
        for line in text.split('\n'):
            if line != "":
                writer.writerow([line])


# read_txt_to_csv(image, "image.")
# read_txt_to_csv(thresholded, "thresholded.")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~` down here is attempts to deal with letters with white inside like ח read as ה
# try1
# import cv2
# import pytesseract
# import numpy as np
# from imutils.perspective import four_point_transform
# image = cv2.imread('POC_sample2_withnoise.tif', 0)
# blurred = cv2.GaussianBlur(image, (5, 5), 0)
# _, thresholded = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# thresholded = cv2.bitwise_not(cv2.floodFill(thresholded, None, (0, 0), 255)[1])
# hsv = cv2.cvtColor(thresholded, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv, (0, 0, 0), (100, 175, 110))
#
# # Morph close to connect individual text into a single contour
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
#
# # Find rotated bounding box then perspective transform
# rect = cv2.minAreaRect(cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])
# warped = four_point_transform(255 - mask, cv2.boxPoints(rect).reshape(4, 2))


# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# try2
# im_in = cv2.imread("nickel.jpg", cv2.IMREAD_GRAYSCALE)
# _, im_out = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#
# im_out = cv2.bitwise_not(cv2.floodFill(im_out, None, (0, 0), 255)[1])
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv, (0, 0, 0), (100, 175, 110))
#
# # Morph close to connect individual text into a single contour
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
#
# # Find rotated bounding box then perspective transform
# rect = cv2.minAreaRect(cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])
# warped = four_point_transform(255 - mask, cv2.boxPoints(rect).reshape(4, 2))


# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# try3
# import cv2
# from imutils.perspective import four_point_transform
#
# image = cv2.imread('POC_sample2_withnoise.tif', 0)
# blurred = cv2.GaussianBlur(image, (5, 5), 0)
# _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cnts, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# rect = cv2.minAreaRect(cnts[0])
# warped = four_point_transform( thresholded, cv2.boxPoints(rect).reshape(4, 2))
# cv2.imshow("image",warped)
# cv2.waitKey(0)
# read_txt_to_csv(warped, "testnewcode")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DATABASE_PATH = r"C:\Users\yarin\PycharmProjects\DHC\GnazimProject\כרטסת עיתונות, גנזים - א - ל\ח - ל"
import re


def find_four_digit_substring(string):
    pattern = r'[1-2]?\d{3}-[1-2]?\d{3}'
    match = re.search(pattern, string)
    if match:
        return match.group()
    return ""


def find_cd_label(string):
    pattern = r'(CD|D)?00\d+?\w?\s?'
    match = re.search(pattern, string)
    if match:
        return match.group()
    return ""


# this code travsl the folder in DB and can open the images even if its english and hebrew
# todo: 1. open an Excel
#  file 2. break the file path name to column data
#  3. add the text that was read from the image to a text column
#  4. count number of images in each folder name
data_col_names = ['Identifier', 'Path', 'File Name', 'Author Subject', 'Type', "Author Type Count", 'Years',
                  'Scanned Text']
data = pd.DataFrame(columns=data_col_names)
problem_folders = []
cnt = 0


def extract_text(path):
    # How to load image with english and hebrew path:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ocr_text = re.sub(r'\n+', ' ', pytesseract.image_to_string(thresholded, lang='heb'))
    return ocr_text


for dirpath, foldernames, filesname in os.walk(DATABASE_PATH):  # search in database folder
    if len(filesname) > 0:
        index = dirpath.find("GnazimProject")
        if index != -1:
            new_data = dict.fromkeys(data_col_names, "")
            new_data["Author Type Count"] = len(filesname)
            new_data["Path"] = dirpath[index + len("GnazimProject") + 1:]
            preprocessed_dirpath = dirpath[index + len("GnazimProject") + 1:].split("\\")
            preprocessed_dirpath.pop(0)
            for i, string in enumerate(reversed(preprocessed_dirpath)):
                preprocessed_str = string.replace(find_cd_label(string), '')
                years = find_four_digit_substring(preprocessed_str)
                preprocessed_str = preprocessed_str.replace(years, '')
                if new_data["Years"] == "" and years != "":
                    new_data["Years"] = years
                if len(preprocessed_str) > 2 and "," in preprocessed_str and new_data["Type"] == "" and i == 0:
                    if preprocessed_str.count("-") > 0:
                        split_strings = [s.strip() for s in preprocessed_str.split("-", 1)]
                        new_data["Type"] = next(s for s in split_strings if "," not in s)
                        new_data["Author Subject"] = next(
                            s for s in split_strings if s != new_data["Type"] and "," in s)
                    elif new_data["Author Subject"] == "":
                        new_data['Author Subject'] = preprocessed_str.strip()
            if new_data['Author Subject'] == "":
                for string in foldernames:
                    if string.count("-") > 0 and new_data['Author Subject'] == "":
                        new_data["Author Subject"] = [s.strip() for s in string.split("-", 1)].pop(0)
                    # if "-" in string and new_data['Author Subject'] == "":
                    #     new_data['Author Subject'] = [s.strip() for s in string.split("-")][0]
            for name in filesname:
                if name.endswith("tif"):
                    cnt += 1
                    new_data["Identifier"] = cnt
                    new_data["File Name"] = name

                    image_path = new_data["Path"] + "\\" + new_data["File Name"]
                    new_data["Scanned Text"] = extract_text(image_path)

                    df = pd.DataFrame(new_data, index=[0])  # specify the index explicitly
                    data = data.append(df, ignore_index=True)  # assign the updated dataframe to data
        else:
            problem_folders.append(dirpath)
    a = 3

# with pd.ExcelWriter('GnazimDB.xlsx') as writer:
#      data.to_excel(writer, index=False, sheet_name='Sheet1')
