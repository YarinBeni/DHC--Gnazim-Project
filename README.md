## Project Gnazim

## Description
The Ganazim Institute is a literary research institute that contains the largest archive of Hebrew literature in the world.
The archive contains about 800 archival collections of writers, poets, playwrights, thinkers, editors, and writers from the end of the 19th century to the present day. 
The archive includes manuscripts, letters, various personal documents, photos, and a unique recording collection.

<p float="left">
  <img src="https://github.com/YarinBeni/DHC--Gnazim-Project/blob/main/data_images_examples/POC_sample2_withnoise%20(1).png?raw=true" width="350" height="250" alt="Noisy OCR Sample Image" />
  <img src="https://github.com/YarinBeni/DHC--Gnazim-Project/blob/main/data_images_examples/POC_sample3_handwriten.png?raw=true" width="350" height="250" alt="Handwritten OCR Sample Image" />
</p>

*Figures: Displayed above are two contrasting examples of the data from the Ganazim Institute's archive. The first image is a relatively clean document, reflecting the best-case scenario for text extraction via OCR. The second is a handwritten document with a higher degree of complexity, indicative of the varied handwriting styles present in the collection.*



## Project Goals
The goal of the project is to extract metadata and Hebrew text from the 180,000 scanned images in the Ganazim Institute archive and make it accessible for research and work.

## Methods
In this project, we have utilized a number of methods to extract and analyze Hebrew text from scanned images of historical documents in the Ganazim Institute archive. Our primary tool for extracting text from these images is the OCR Pytesseract model in Python, which is able to accurately recognize Hebrew characters.
To improve the accuracy of our OCR model, we have employed various image processing techniques using OpenCV, including GaussianBlur and Otsu thresholding. These methods help to enhance the quality of the scanned images and ensure that the text is extracted as accurately as possible.
In addition to extracting the text from the scanned images, we have also used a pandas dataframe as a data structure for extracting metadata from the Google Cloud folder structures and names. This has allowed us to efficiently organize and analyze the data from the archive, and create a CSV file containing all of the extracted data.
Overall, the combination of OCR, image processing, and data analysis methods has allowed us to efficiently extract and analyze Hebrew text from the scanned images in the Ganazim Institute archive, providing valuable insights into the history and culture of the Jewish people.

## How to Install

1. Ensure you are in the correct project directory.
2. Run the following command in your terminal to install the dependencies:

```shell
pip install -r requirements.txt
```

## Pseudo Code Logic of main.py

1. **GCP Connection**:
   - Initialize GCP Connection.

2. **Queue Initialization**:
   - Initialize Queue with Root Folder ID.
   - Set Reconnection Threshold for GCP.
   - Set File Processing Count Threshold.
   - Set Queue Save and Read Threshold.

3. **Main Processing Loop**:
   - While Queue is not empty:
     - Dequeue a Folder from Queue.
     - Retrieve Data of Previously Processed and Problematic Files from Local Storage.
     - Count Already Processed Files and Create Set of Processed File Paths.
     - Initialize Counter for Files Processed in Current Run.

     - If Reconnection Threshold is Reached, Reconnect to GCP.

     - **File Processing**:
       - Get List of Files in Current Folder.
       - Initialize DataFrame for New Processed Files.
       - For Each File in Folder:
         - If File is an Image and Not Already Processed:
           - Extract Meta Data from File Path (gcp_extract_years_author_type).
           - Download File from GCP.
           - Process Image to Detect Relevant Paragraphs (ImageProcessor).
           - Extract Text from Detected Paragraphs using GCP OCR (gcp_extract_text_from_image).
           - Append Processed File Data to New DataFrame.
           - If GCP Connection Error Occurs:
             - Attempt Reconnection (gcp_reconnect) and Reprocess File.
             - If Reconnection Fails, Log Error and Update Problematic Files Data in Local Storage, Break from Loop.
         - Else If File is Not an Image, Add File (Folder) to Queue for Later Processing.

     - **Post-Processing**:
       - If Folder is Successfully Processed, Append New Data to DataFrame of Processed Files.
       - Update Local CSV Files with New Processed and Problematic Files Data.
       - If Queue Save and Read Threshold is Reached, Read Next Set of Folders from Saved Queue State.
       - Perform Function Profiling for Each Function Call.

4. **Finalization**:
   - After Queue is Empty:
     - Generate GCP File and Folder Links using create_df_gcp_file_links.
     - Append Links to Processed Files DataFrame.
     - Save Updated DataFrame to Local Storage.
   - Update Profiling Data with Run Timestamp at End of Run.
   - Perform Overall Time Analysis*.

## Future Goals
In the future, we plan to classify handwritten and typed text and build a dataset suitable for research purposes. 
This dataset can be used to study Hebrew literature and its evolution over time. We also plan to explore other machine learning models and techniques that can improve the accuracy and efficiency of our text extraction methods.

