## Project Gnazim

## Description
The Ganazim Institute is a literary research institute that contains the largest archive of Hebrew literature in the world.
The archive contains about 800 archival collections of writers, poets, playwrights, thinkers, editors, and writers from the end of the 19th century to the present day. 
The archive includes manuscripts, letters, various personal documents, photos, and a unique recording collection.

## Project Goals
The goal of the project is to extract metadata and Hebrew text from the 180,000 scanned images in the Ganazim Institute archive and make it accessible for research and work.

## Methods
In this project, we have utilized a number of methods to extract and analyze Hebrew text from scanned images of historical documents in the Ganazim Institute archive. Our primary tool for extracting text from these images is the OCR Pytesseract model in Python, which is able to accurately recognize Hebrew characters.
To improve the accuracy of our OCR model, we have employed various image processing techniques using OpenCV, including GaussianBlur and Otsu thresholding. These methods help to enhance the quality of the scanned images and ensure that the text is extracted as accurately as possible.
In addition to extracting the text from the scanned images, we have also used a pandas dataframe as a data structure for extracting metadata from the Google Cloud folder structures and names. This has allowed us to efficiently organize and analyze the data from the archive, and create a CSV file containing all of the extracted data.
Overall, the combination of OCR, image processing, and data analysis methods has allowed us to efficiently extract and analyze Hebrew text from the scanned images in the Ganazim Institute archive, providing valuable insights into the history and culture of the Jewish people.

## Code Logic of main.py

The script follows the outlined logic to process folders and files in Google Cloud Platform (GCP):

1. **GCP Connection**:
   - Establish a connection to GCP.
   
2. **Initialization**:
   - Create a stack initialized with the root folder to be processed.
   - Set a threshold for reconnection to GCP.

3. **Main Processing Loop**:
   - Continue processing until the folder stack is empty.
   - Dequeue a folder from the stack.
   - Retrieve locally saved data of previously processed files.
   - Determine the count of files already processed.
   - Create a dictionary to track processed files using data from local storage.
   - Retrieve locally saved data of problematic folders.
   - Initialize a counter for the number of files processed in the current run to 0.

   - **Reconnection Check**:
     - Check if the reconnection threshold is reached, and if yes, reconnect to GCP.

   - **File Processing**:
     - Obtain a list of files from the current folder.
     - Initialize a temporary data table to store newly processed files' data.
     - Loop through all files in the current folder:
       - For image files that haven't been processed yet:
         - Attempt to process the file and append new row data to the running data table.
         - Handle GCP connection errors by refreshing the GCP token and reattempting file processing.
           - If the second attempt fails, exit the loop and raise an error indicating the folder processing failure, along with relevant error and folder information.
           - If the second attempt succeeds, append new row data to the running data table and continue to the next file.
         - For non-GCP connection errors, log the folder name and error details, update the problematic folders data on local disk, and exit the loop.
       - For non-image files (likely other folders), add the file (folder) to the stack for processing.

   - **Post-Processing**:
     - If the entire folder was successfully processed, append the running data table to the locally saved data of processed files.


## Future Goals
In the future, we plan to classify handwritten and typed text and build a dataset suitable for research purposes. 
This dataset can be used to study Hebrew literature and its evolution over time. We also plan to explore other machine learning models and techniques that can improve the accuracy and efficiency of our text extraction methods.

