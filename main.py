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

# def read_txt_to_csv(im, filename):
#     global writer, line
#     text = pytesseract.image_to_string(im, lang='heb')
#
#     print(text, "\n")
#     with open(filename + 'output.csv', 'w', newline='') as th_csvfile:
#         writer = csv.writer(th_csvfile)
#         for line in text.split('\n'):
#             if line != "":
#                 writer.writerow([line])


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
DATABASE_PATH = r"C:\Users\yarin\PycharmProjects\DHC\GnazimProject\כרטסת עיתונות, גנזים - א - ל"
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

def extract_text(path):
    # How to load image with english and hebrew path:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ocr_text = re.sub(r'\n+', ' ', pytesseract.image_to_string(thresholded, lang='heb'))
    return ocr_text


def get_count(df):
    if len(df) == 0:
        return 0
    else:
        return df['Identifier'].max()


def extract_years_auther_type(preprocessed_dirpath, foldernames):
    years, author_subject, type_subject = "", "", ""
    for i, string in enumerate(reversed(preprocessed_dirpath)):
        preprocessed_str = string.replace(find_cd_label(string), '')
        years_candidate = find_four_digit_substring(preprocessed_str)
        if years == "" and years_candidate != "":
            years = years_candidate

        preprocessed_str = preprocessed_str.replace(years, '')
        if len(preprocessed_str) > 1 and "," in preprocessed_str and type_subject == "" and i == 0:
            if preprocessed_str.count("-") > 0:
                split_strings = [s.strip() for s in preprocessed_str.split("-", 1)]
                type_subject = next(s for s in split_strings if "," not in s)
                author_subject = next(s for s in split_strings if s != type_subject and "," in s)
            elif author_subject == "":
                author_subject = preprocessed_str.strip()
    if author_subject == "":
        for string in foldernames:
            if string.count("-") > 0 and author_subject == "":
                author_subject = [s.strip() for s in string.split("-", 1)].pop(0)
            # if "-" in string and new_data['Author Subject'] == "":
            #     new_data['Author Subject'] = [s.strip() for s in string.split("-")][0]
    return years, author_subject, type_subject


data_col_names = ['Identifier', 'Path', 'File Name', 'Author Subject', 'Type', "Author Type Count",
                  'Years', 'Scanned Text']


def process_folder(dirpath, foldernames, filesname):
    """
    Process the files in a folder and add them to the dataframe
    """
    unsuccessful_folder = ""
    if len(filesname) > 0:
        try:
            index = dirpath.find("GnazimProject")
            if index != -1:
                data = folder_to_df(dirpath, filesname, foldernames, index)
                with pd.ExcelWriter('YarinGnazimDB.xlsx') as writer:
                    data.to_excel(writer, index=False, sheet_name='Sheet1')
            else:
                raise Exception(f"~~~~~~~\nError! Problem in folder:\n{dirpath}\n~~~~~~~")
        except:
            unsuccessful_folder = dirpath
    return unsuccessful_folder


def folder_to_df(dirpath, filesname, foldernames, index):
    data = get_data()
    new_data = dict.fromkeys(data_col_names, "")
    processed_path = dirpath[index + len("GnazimProject") + 1:].split("\\")
    processed_path.pop(0)
    new_data["Years"], new_data["Author Subject"], new_data["Type"] = extract_years_auther_type(
        processed_path,
        foldernames)
    new_data["Author Type Count"] = len(filesname)
    new_data["Path"] = dirpath[index + len("GnazimProject") + 1:]
    for name in filesname:
        if name.endswith("tif"):
            cnt = get_count(data)
            cnt += 1
            new_data["Identifier"] = cnt
            new_data["File Name"] = name
            image_path = new_data["Path"] + "\\" + new_data["File Name"]
            new_data["Scanned Text"] = extract_text(image_path)
            new_row = pd.DataFrame(new_data, index=[0])  # specify the index explicitly
            data = pd.concat([data, new_row], axis=0)  # assign the updated dataframe to data
    return data


def get_data():
    if os.path.isfile("YarinGnazimDB.xlsx"):
        df = pd.read_excel("YarinGnazimDB.xlsx")
        df.fillna('', inplace=True)
    else:
        df = pd.DataFrame(columns=data_col_names)
    return df


# def get_problem_data():
#     if os.path.isfile("YarinProblemsGnazimDB.xlsx"):
#         df = pd.read_excel("YarinProblemsGnazimDB.xlsx")
#     else:
#         df = pd.DataFrame(columns=["Problem Folders Names "])
#     return df


def process_database_folder(path):
    """
    Process the files in the database folder
    """
    problem_folders = []
    folder_cnt = 0
    prob_cnt = 0
    for dirpath, foldernames, filesname in os.walk(path):
        new_problem_folder = process_folder(dirpath, foldernames, filesname)
        print(f"\n~~~Finished Folder Number {folder_cnt} , Name {dirpath}!~")
        folder_cnt += 1
        if new_problem_folder:
            print(f"\nXXX Problem Folder Number {prob_cnt} , Name {dirpath}!~")
            prob_cnt += 1
            problem_folders.append(new_problem_folder)
    problem_df = pd.DataFrame(problem_folders, columns=["Problem Folders Names "], index=[0])
    with pd.ExcelWriter('YarinProblemsGnazimDB.xlsx') as writer:
        problem_df.to_excel(writer, index=False, sheet_name='Sheet1')
    print("\n~process_database_folder DONE!~")


# process_database_folder(DATABASE_PATH)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Gnazim Project projectid: gnazim-project
import urllib.request

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from PIL import Image
folder_id = "1sciJWxtjAxbas-fyxuIiXPANOXGxectN"
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication.
drive = GoogleDrive(gauth)

dff_col_names = ['Path', 'File id']
dff = pd.DataFrame(columns=dff_col_names)


def gcp_extract_text(data):
    file_id = data["File id"]
    # Download the file using PyDrive2
    file = drive.CreateFile({'id': file_id})
    file.GetContentFile("testname")
    # Load the downloaded file into a cv2 image in grayscale
    img_cv2 = cv2.imread("testname", cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur and thresholding
    blurred = cv2.GaussianBlur(img_cv2, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform OCR and return the extracted text
    ocr_text = re.sub(r'\n+', ' ', pytesseract.image_to_string(thresholded, lang='heb'))
    return ocr_text




def gcp_extract_years_auther_type(preprocessed_dirpath):
    years, author_subject, type_subject = "", "", ""
    for i, string in enumerate(reversed(preprocessed_dirpath)):
        preprocessed_str = string.replace(find_cd_label(string), '')
        years_candidate = find_four_digit_substring(preprocessed_str)
        if years == "" and years_candidate != "":
            years = years_candidate

        preprocessed_str = preprocessed_str.replace(years, '')
        if len(preprocessed_str) > 1 and "," in preprocessed_str and type_subject == "" and i == 0:
            if preprocessed_str.count("-") > 0:
                split_strings = [s.strip() for s in preprocessed_str.split("-", 1)]
                type_subject = next(s for s in split_strings if "," not in s)
                author_subject = next(s for s in split_strings if s != type_subject and "," in s)
            elif author_subject == "":
                author_subject = preprocessed_str.strip()

    return years, author_subject, type_subject


def gcp_folder_to_df(new_data):
    big_data = get_data()
    processed_path = new_data["Path"].split("\\")
    processed_path.pop(0)
    new_data["Years"], new_data["Author Subject"], new_data["Type"] = gcp_extract_years_auther_type(
        processed_path)

    cnt = get_count(big_data)
    cnt += 1
    new_data["Identifier"] = f"IDGNAZIM000{cnt}"
    new_data["Scanned Text"] = gcp_extract_text(new_data)
    new_row = pd.DataFrame(new_data, index=[0])  # specify the index explicitly
    big_data = pd.concat([big_data, new_row], axis=0)  # assign the updated dataframe to data
    return big_data


def print_files_in_folder(folder_id, folder_title=""):
    # search for files in the folder
    file_list = drive.ListFile({'q': "'%s' in parents and trashed=false" % folder_id}).GetList()
    sample_cnt = 0
    # print the name and ID of each file in the folder that ends with .tif
    for file1 in file_list:
        current_folder_title = folder_title + "\\" + file1['title']
        if file1['title'].endswith('.tif'):
            sample_cnt += 1
            # print('directory path: %s ----- id: %s' % (current_folder_title, file1['id']))
            new_data = dict.fromkeys(dff_col_names, "")
            new_data["Path"] = current_folder_title
            new_data["File id"] = file1['id']
            # new_row = pd.DataFrame(new_data, index=[0])
            # Extracting the actual folder ID and actual file name and saving them in the new_data dictionary
            new_data["actual_folder_id"] = file1['parents'][0]['id']
            new_data["actual_file_name"] = file1['title']
            print(f"File Path: {current_folder_title}\nFile Title: {file1['title']}\n")
            new_d = gcp_folder_to_df(new_data)
            a = 3


        else:
            print()
            print()
            print()
            print_files_in_folder(file1['id'], current_folder_title)


print_files_in_folder(folder_id, "")


# todo: Now the code is able to traverse cloud, download image, ocr it, and fill newdata as before, now need to:
#  1) move get_data call to main function and adjust functions maybe as a class and class var 
#  2) need to write the newdata into the updated csv table
#  3) to do try expect as in the local case code so if some artibutes will fail to record problem files and folders.


# todo: 1) make the code oop to fix the save to dff problem in the GCP code
#  2) finish the code for GCP and create a full csv for dataset
#  3) refine ocr and parsed text in csv with implemented algorithms in open refine such as edit distance etc
#  4) find way to improve ocr with bagging with other models , need to set a meeting with dicta people
#  5) find way to classify handwriting and typed (models probabilties? length of word? number of symbolys insted of letters?)
#  6) create an sql database online?
#  *) test performance of ocr auther name with the label from the a confusion matrix
#  important for me to know from yael: 1) what is the timeline of the project, 2) what is the project goal, 3) can we get a paper from it ?
