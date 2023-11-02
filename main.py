# Gnazim Project projectid: gnazim-project

import os
import cv2
import pandas as pd
from ParagraphDetector import ImageProcessor
import re
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import time
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Code logic:
# Connect to GCP
# Initialize a stack with the root folder to process
# Initialize a threshold for reconnection to GCP

# Loop until the folder stack is empty:
#     Dequeue a folder from the stack
#     Retrieve locally saved data of previously processed files
#     Determine the count of files already processed
#     Create a dictionary to track processed files using data from local storage
#     Retrieve locally saved data of problematic folders
#     Initialize a counter for the number of files processed in the current run to 0

#     Check if the reconnection threshold is reached:
#         If yes, reconnect to GCP

#     # Process Files in the Current Folder
#     Obtain a list of files from the current folder
#     Initialize a temporary data table to store newly processed files' data

#     Loop through all files in the current folder:
#         If the file is an image and hasn't been processed yet:
#             Attempt to process the file and append new row data to the running data table
#             If a GCP connection error occurs:
#                 Attempt to refresh the GCP token and reprocess the file
#                 If the second attempt fails:
#                     Exit the loop and raise an error indicating the folder processing failure,
#                     along with relevant error and folder information
#                 Else if the second attempt succeeds:
#                     Append new row data to the running data table and continue to the next file
#             If a non-GCP connection error occurs:
#                 Log the folder name and error details
#                 Update the problematic folders data on local disk and exit the loop
#         Else if the file is not an image (likely another folder):
#             Add the file (folder) to the stack for processing

#     If the entire folder was successfully processed:
#         Append the running data table to the locally saved data of processed files

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ************************************************************************************************************************

data_col_names = ['identifier', 'path', 'gcp_file_id', 'folder_name', 'author_subject', 'type',
                  'gcp_folder_id', 'file_name', 'ocr_writen_on', 'ocr_writen_by',
                  'ocr_main_content', 'ocr_additional_content', 'ocr_writen_on_coords',
                  'paragraphs_detection_successes', 'ocr_all_text_preprocess', 'ocr_all_text_no_preprocess',
                  'ocr_main_content_all_text_preprocess', 'ocr_main_content_all_text_no_preprocess']
problem_col_names = ['identifier', 'path', 'gcp_file_id', 'folder_name', 'author_subject', 'type',
                     'gcp_folder_id', 'file_name', 'ocr_writen_on', 'ocr_writen_by',
                     'ocr_main_content', 'ocr_additional_content', 'ocr_writen_on_coords',
                     'paragraphs_detection_successes', 'ocr_all_text_preprocess', 'ocr_all_text_no_preprocess',
                     'ocr_main_content_all_text_preprocess', 'ocr_main_content_all_text_no_preprocess',
                     "error_message"]

file_extensions = [
    '.txt', '.doc', '.docx', '.pdf', '.rtf',
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',
    '.mp3', '.wav', '.ogg', '.flac',
    '.mp4', '.avi', '.mov', '.wmv', '.mkv',
    '.csv', '.xml', '.json', '.sql', '.db',
    '.xls', '.xlsx', '.ods',
    '.ppt', '.pptx', '.odp',
    '.html', '.htm', '.css', '.js',
    '.zip', '.rar', '.tar', '.gz',
    '.exe', '.dll', '.sys',
    '.py', '.java', '.c', '.cpp', '.h'
]


def get_data(problem_files=False):
    folder_name = "results"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    csv_file_path = 'results/gnazim_db_meta_data.csv'
    col_names = data_col_names  # Assuming data_col_names is defined elsewhere in your code
    if problem_files:
        csv_file_path = 'results/gnazim_db_problem_files.csv'
        col_names = problem_col_names  # Assuming problem_col_names is defined elsewhere in your code
    # Removed the duplicated condition
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        df.fillna('', inplace=True)
    else:
        df = pd.DataFrame(columns=col_names)
    return df


def establish_connection():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)
    # # Check if the token is expired


def find_four_digit_substring(string):
    # current year, can be manually set or automatically taken from current time
    current_year = 2023

    # Pattern based on cases
    patterns = [
        r'(?<!\d)(1?\d{3})-(1?\d{3})(?!\d)',  # 1a
        r'(?<!\d)2[0][0-2][0-3]-(1?\d{3})(?!\d)|(?<!\d)(1?\d{3})-2[0][0-2][0-3](?!\d)',  # 1b
        r'(?<!\d)2[0][0-2][0-3]-2[0][0-2][0-3](?!\d)',  # 1c
        r'(?<!\d)(1?\d{3})(?!\d)',  # 2a
        r'(?<!\d)2[0][0-2][0-3](?!\d)'  # 2b
    ]

    for pattern in patterns:
        match = re.search(pattern, string)
        if match:
            year_group = match.group(0)
            # Filter out matches exceeding the current year
            years = [int(y) for y in year_group.split('-')]
            if all(year <= current_year for year in years):
                return year_group
    return ""


def find_cd_label(string):
    pattern = r'(CD|D)?00\d+?\w?\s?'
    match = re.search(pattern, string)
    if match:
        return match.group()
    return ""


def is_path_processed(existing_data, path_to_check):
    """
    Check if a given path is already present in the existing data DataFrame.

    :param existing_data: DataFrame of the already processed data.
    :param path_to_check: The path of the file to check.
    :return: Boolean value indicating if the path is already processed.
    """
    return path_to_check in existing_data["path"].values


def is_image(file_name):
    return file_name.endswith('.tif')


def gcp_extract_text_from_image(data, drive):
    file_id = data["gcp_file_id"]
    # Download the file using PyDrive2
    file = drive.CreateFile({'id': file_id})
    file.GetContentFile("testname")
    # Load the downloaded file into a cv2 image in grayscale
    img = cv2.imread("testname", cv2.IMREAD_GRAYSCALE)
    processor = ImageProcessor(img)
    ocr_meta_data = processor.run()
    return ocr_meta_data


def gcp_extract_years_auther_type(preprocessed_dirpath):
    years, author_subject, type_subject = "", "", ""
    for i, string in enumerate(reversed(preprocessed_dirpath)):
        preprocessed_str = string.replace(find_cd_label(string), '')
        years_candidate = find_four_digit_substring(preprocessed_str)
        if years == "" and years_candidate != "" and len(years_candidate) > 3:
            years = years_candidate

        preprocessed_str = preprocessed_str.replace(years, '')
        if len(preprocessed_str) > 1 and "," in preprocessed_str and type_subject == "" and i <= 1:
            if preprocessed_str.count("-") > 0:
                split_strings = [s.strip() for s in preprocessed_str.split("-", 1)]
                type_subject = next((s for s in split_strings if "," not in s), "")
                author_subject = next((s for s in split_strings if s != type_subject and "," in s), "")
            elif author_subject == "":
                author_subject = preprocessed_str.strip()

    return years, author_subject, type_subject


def gcp_folder_to_df(new_data, drive, cnt):
    processed_path = new_data["path"].split("\\")
    processed_path.pop(0)

    new_data["years"], new_data["author_subject"], new_data["type"] = gcp_extract_years_auther_type(
        processed_path)

    new_data["identifier"] = f"IDGNAZIM000{cnt}"
    ocr_meta_data = gcp_extract_text_from_image(new_data, drive)
    new_data.update(ocr_meta_data)  # Update new_data dictionary with ocr_meta_data
    new_row = pd.DataFrame(new_data, index=[0])  # specify the index explicitly

    print("-" * 110)
    print(f"-----------Meta data for file number {cnt} is:"
          f"----------------------------------------------------------------")
    # for k in new_data.keys():
    #     print(k, ": ", new_data[k])
    print("-" * 110)
    print()
    big_data = pd.concat([get_data(), new_row], axis=0)  # assign the updated dataframe to data
    return big_data


# Function to check if a string does not end with any of the listed file extensions
def is_not_file(string):
    return not any(string.endswith(ext) for ext in file_extensions)


def get_count(df):
    if len(df) == 0:
        return 0
    else:
        return df.shape[0]


def gcp_process_files_in_folder(folder_id, drive, scanned_files_amount_now, folder_title=""):
    folder_files_list = drive.ListFile({'q': "'%s' in parents and trashed=false" % folder_id}).GetList()
    problem_cnt = get_count(get_data(problem_files=True))
    sample_cnt = scanned_files_amount_now
    current_data = get_data()
    # List to hold folder parameters for non-image files
    folders_to_process = []
    new_processed_files = pd.DataFrame(columns=data_col_names)
    start_time = time.time()
    print(f"Start Attempt to Process Folder: {folder_title}")
    i = 0
    for file in folder_files_list:
        i += 1
        try:
            current_folder_title = folder_title + "\\" + file['title']
            new_row = []

            if is_image(file['title']):
                if not is_path_processed(current_data, current_folder_title):
                    print(f"Process file:{file['title']} First attempt Successfully")
                    new_row = gcp_process_file(current_folder_title, drive, file, sample_cnt, folder_title)
                    sample_cnt += 1
                    new_processed_files = pd.concat([new_processed_files, new_row], axis=0)
            elif is_not_file(file['title']):
                folders_to_process.append((file['id'], current_folder_title))

        except Exception as e:
            if str(e) == "invalid_grant: Bad Request":
                try:
                    print(f"Process file:{file} First attempt failed! Connection Problem, Starting Second Attempt...")
                    gcp_reconnect(drive)
                    new_row = gcp_process_file(current_folder_title, drive, file, sample_cnt, folder_title)
                    sample_cnt += 1
                    new_processed_files = pd.concat([new_processed_files, new_row], axis=0)
                    continue
                except:
                    print(f"ReProcess file:{file} Second attempt failed! Add To Problems...\n")
            new_row = create_new_problem_row(current_folder_title, e, file, new_row, problem_cnt)
            write_problem_folder_to_csv(new_row)
            raise e
    end_time = time.time()

    if sample_cnt > scanned_files_amount_now:
        new_processed_files.to_csv('results/gnazim_db_meta_data.csv', mode='a', header=False, index=False)

        print(f"Folder {folder_title} Processed Successfully and Save to Disk after {end_time - start_time} time\n")
    else:
        print(
            f"Folder {folder_title} Processed Successfully and No Files Saved to Disk after {end_time - start_time} time\n")
    return folders_to_process


def gcp_reconnect(drive):
    drive.Refresh()
    print(f"Connection attempt failed. Retrying in {60} seconds...")
    time.sleep(60)


def create_new_problem_row(current_folder_title, e, file, new_row, problem_cnt):
    try:
        problem_cnt += 1
        # Handle exceptions by appending the problematic folder and file to 'gnazim_db_problem_folders.csv'
        print(f"Exception number {problem_cnt}!\nencountered for folder: {current_folder_title}\n"
              f"file: {file['title']}\nError: {e}\n")
        new_row.update({"folder_name": [current_folder_title],
                        "file_name": [file['title']],
                        "error_message": str(e)})
    except Exception as error:
        new_row = {key: "" for key in data_col_names}
        new_row.update({"folder_name": [current_folder_title],
                        "file_name": [file['title']],
                        "error_message": str(e)})
    return new_row


def gcp_process_file(file_path, drive, file1, count, folder_title):
    # initialize new row
    new_row = {key: "" for key in data_col_names}

    # Process Meta Data From Folder Structure
    new_row["path"] = file_path
    new_row["gcp_file_id"] = file1['id']
    new_row["folder_name"] = folder_title
    new_row["gcp_folder_id"] = file1['parents'][0]['id']
    new_row["file_name"] = file1['title']
    new_row["Years"] = ""  # this is redundant column we need to remove so in the end of the run will do
    processed_path = new_row["path"].split("\\")
    processed_path.pop(0)
    new_row["years"], new_row["author_subject"], new_row["type"] = gcp_extract_years_auther_type(processed_path)
    count += 1
    new_row["identifier"] = f"IDGNAZIM000{count}"

    # Process OCR Data From Image
    ocr_meta_data = gcp_extract_text_from_image(new_row, drive)
    # print("-" * 110)
    # print(f"-----------Meta data for file number {count} is:"
    #       f"----------------------------------------------------------------")
    # for k in ocr_meta_data.keys():
    #     print(k, ": ", ocr_meta_data[k])
    # print("-" * 110)
    # print()

    # Save All Processed New Row Data
    new_row.update(ocr_meta_data)  # Update new_data dictionary with ocr_meta_data
    new_row = pd.DataFrame(new_row, index=[0])  # specify the index explicitly

    return new_row


def run():
    # GCP Connection
    current_drive = establish_connection()
    # Initialize Folders root
    root_folder_id = "1sciJWxtjAxbas-fyxuIiXPANOXGxectN"  # First Folder to Preprocess Than DFS OR BFS On Tree
    folders_to_process = [(root_folder_id, "")]
    reconnect_trashold = 600
    problem_folders_to_process = []  # Store problematic folders' data here
    data = get_data()
    scanned_files_amount_in_beginning = get_count(data)
    gcp_processed_folders_names = set(data['folder_name'].unique())
    while folders_to_process:

        current_folder_id, current_folder_title = folders_to_process.pop(0)

        data = get_data()
        scanned_files_amount_now = get_count(data)
        print(
            f"\nscanned_files_amount_in_beginning:{scanned_files_amount_in_beginning} , scanned_files_amount_now: {scanned_files_amount_now}")

        scanned_files_amount = scanned_files_amount_now - scanned_files_amount_in_beginning
        if scanned_files_amount > 4000:
            break

        if scanned_files_amount >= reconnect_trashold:  # Reconnect Every 600 read samples
            reconnect_trashold += reconnect_trashold
            gcp_reconnect(current_drive)

        try:
            new_folders_to_process = gcp_process_files_in_folder(current_folder_id, current_drive,
                                                                 scanned_files_amount_now, current_folder_title)
            new_filtered_folders_to_process = [(folder_id, folder_title) for (folder_id, folder_title) in
                                               new_folders_to_process if
                                               folder_title not in gcp_processed_folders_names]
            folders_to_process.extend(new_filtered_folders_to_process)
        except Exception as er:
            if str(er) == "invalid_grant: Bad Request":
                print(f"Error processing folder {current_folder_id} - {current_folder_title}: {er}")
                print("***** Finished Run due to GCP Connection Problem *****")
                break
            else:
                # Log and save other exceptions for later inspection
                print(f"Error processing folder {current_folder_id} - {current_folder_title}: {er}")
                print("***** Continue Run to Different Folder, Problem Been Saved For Future Fix *****")
                # problem_folders_to_process.append(
                #     {"folder_id": current_folder_id, "folder_title": current_folder_title})
                # write_problem_folder_to_excel(problem_folders_to_process)
    print(f"End Of Run:\nProblem folders:\n{problem_folders_to_process}")


def write_problem_folder_to_excel(new_problem_row):
    # If there are problematic folders, save them to Excel
    if new_problem_row:
        new_problem_data = pd.DataFrame(new_problem_row)

        problem_folders = get_data(problem_files=True)
        if problem_folders.shape[0] == 0:
            new_problem_folders = new_problem_data
        else:
            new_problem_folders = pd.concat([problem_folders, new_problem_data], axis=0, ignore_index=True)

        with pd.ExcelWriter('results/gnazim_db_problem_files.xlsx') as writer:
            new_problem_folders.to_excel(writer, index=False, sheet_name='Sheet1')


def write_problem_folder_to_csv(new_problem_row):
    # If there are problematic folders, save them to CSV
    if new_problem_row:
        new_problem_data = pd.DataFrame(new_problem_row)
        csv_file_path = 'results/gnazim_db_problem_files.csv'

        # Check if the CSV file already exists to decide whether to write headers
        write_header = not os.path.exists(csv_file_path)

        # Append the new data to the CSV file (or create a new file if it doesn't exist)
        new_problem_data.to_csv(csv_file_path, mode='a', index=False, header=write_header)


# ************************************************************************************************************************


if __name__ == "__main__":
    run()
