## Project Gnazim

## Description
The Ganazim Institute is a literary research institute that contains the largest archive of Hebrew literature in the world.
The archive contains about 800 archival collections of writers, poets, playwrights, thinkers, editors, and writers from the end of the 19th century to the present day. 
The archive includes manuscripts, letters, various personal documents, photos, and a unique recording collection.

## Project Goals
The goal of the project is to extract metadata and Hebrew text from the 180,000 scanned images in the Ganazim Institute archive. We aim to achieve this using OCR Pytesseract model in Python, along with NLP and image processing methods.

## Methods
We are using OCR Pytesseract model in Python, along with NLP and image processing methods, to extract metadata and Hebrew text from the scanned images in the Ganazim Institute archive.

## Future Goals
In the future, we plan to classify handwritten and typed text and build a dataset suitable for research purposes. 
This dataset can be used to study Hebrew literature and its evolution over time. We also plan to explore other machine learning models and techniques that can improve the accuracy and efficiency of our text extraction methods.

## Methods
In this project, we have utilized a number of methods to extract and analyze Hebrew text from scanned images of historical documents in the Ganazim Institute archive. Our primary tool for extracting text from these images is the OCR Pytesseract model in Python, which is able to accurately recognize Hebrew characters.
To improve the accuracy of our OCR model, we have employed various image processing techniques using OpenCV, including GaussianBlur and Otsu thresholding. These methods help to enhance the quality of the scanned images and ensure that the text is extracted as accurately as possible.
In addition to extracting the text from the scanned images, we have also used a pandas dataframe as a data structure for extracting metadata from the Google Cloud folder structures and names. This has allowed us to efficiently organize and analyze the data from the archive, and create a CSV file containing all of the extracted data.
Overall, the combination of OCR, image processing, and data analysis methods has allowed us to efficiently extract and analyze Hebrew text from the scanned images in the Ganazim Institute archive, providing valuable insights into the history and culture of the Jewish people.
