# -*- coding: utf-8 -*-

# https://github.com/NNLP-IL/Hebrew-Resources#optical-character-recognition-ocr


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ USE LLM TO IMPROVE TEXT  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# from transformers import AutoModelForMaskedLM, AutoTokenizer


#
# tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert')
# model = AutoModelForMaskedLM.from_pretrained('dicta-il/dictabert')
#
# model.eval()
#
# sentence = 'בשנת 1948 השלים אפרים קישון את [MASK] בפיסול מתכת ובתולדות האמנות והחל לפרסם מאמרים הומוריסטיים'
#
# output = model(tokenizer.encode(sentence, return_tensors='pt'))
# # the [MASK] is the 7th token (including [CLS])
# import torch
# top_2 = torch.topk(output.logits[0, 7, :], 2)[1]
# print('\n'.join(tokenizer.convert_ids_to_tokens(top_2))) # should print מחקרו / התמחותו

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#NO PREPRO 'צלף עלהנ"ל(מספרות מעריב,ספרות.כת אייר .תש '
#PrePRO '- ;- אל-סיי,חנה כפירא,רות בזרועוה הצלף לֶל הנ ל(מספרות העולם) מעריב,ספרוה,כת אייר תשמו6.6.!1%986 עמ285 / - | '

# from transformers import pipeline
# # https://huggingface.co/dicta-il/dictabert-heq
#
# oracle = pipeline('question-answering', model='dicta-il/dictabert-heq')
#
#
# context = 'בניית פרופילים של משתמשים נחשבת על ידי רבים כאיום פוטנציאלי על הפרטיות. מסיבה זו הגבילו חלק מהמדינות באמצעות חקיקה את המידע שניתן להשיג באמצעות עוגיות ואת אופן השימוש בעוגיות. ארצות הברית, למשל, קבעה חוקים נוקשים בכל הנוגע ליצירת עוגיות חדשות. חוקים אלו, אשר נקבעו בשנת 2000, נקבעו לאחר שנחשף כי המשרד ליישום המדיניות של הממשל האמריקאי נגד השימוש בסמים (ONDCP) בבית הלבן השתמש בעוגיות כדי לעקוב אחרי משתמשים שצפו בפרסומות נגד השימוש בסמים במטרה לבדוק האם משתמשים אלו נכנסו לאתרים התומכים בשימוש בסמים. דניאל בראנט, פעיל הדוגל בפרטיות המשתמשים באינטרנט, חשף כי ה-CIA שלח עוגיות קבועות למחשבי אזרחים במשך עשר שנים. ב-25 בדצמבר 2005 גילה בראנט כי הסוכנות לביטחון לאומי (ה-NSA) השאירה שתי עוגיות קבועות במחשבי מבקרים בגלל שדרוג תוכנה. לאחר שהנושא פורסם, הם ביטלו מיד את השימוש בהן.'
# question = 'כיצד הוגבל המידע שניתן להשיג באמצעות העוגיות?'
#
# oracle(question=question, context=context)
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ USE CONFUSION MATRIX TO IMPROVE TEXT  ~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
import numpy as np
from collections import Counter
from main import get_data


# preprocess
df = get_data()
# Assuming 'df' is your original DataFrame
# Rename columns
df_renamed = df.rename(columns={'author_subject': 'GT',
                                'ocr_writen_on': 'OCR_text',
                                'ocr_all_text_preprocess': 'OCR_to_fix'})
# Filter rows where GT is "גורי, חיים" for POC
# df_filtered = df_renamed[df_renamed['GT'] == " גורי, חיים"]

df_filtered = df_renamed
df_filtered = df_filtered[["GT","OCR_text","OCR_to_fix"]]
df_filtered = df_filtered[(df_filtered['GT'] != "") & (df_filtered['OCR_text'] != "")]

import Levenshtein

# Assuming df_filtered is your DataFrame
def calculate_levenshtein(row):
    return Levenshtein.distance(row['GT'], row['OCR_text'])

df_filtered['levenshtein_distance'] = df_filtered.apply(calculate_levenshtein, axis=1)

df_filtered_high_distance = df_filtered[df_filtered['levenshtein_distance'] < 6]
df_filtered_equal_length = df_filtered[df_filtered['GT'].str.len() == df_filtered['OCR_text'].str.len()]

df_filtered_equal_length_and_distance = df_filtered_equal_length[df_filtered['levenshtein_distance'] < 6]

# Intiliazie Confusion matrix
hebrew_alphabet = 'אבגדהוזחטיכלמנסעפצקרשת'
numbers = '0123456789'
symbols = ', '
all_chars = hebrew_alphabet + numbers + symbols
confusion_matrix = pd.DataFrame(np.zeros((len(all_chars), len(all_chars))), index=list(all_chars), columns=list(all_chars))


# Function to update confusion matrix
def update_confusion_matrix(gt, ocr, matrix):
    for g_char, o_char in zip(gt, ocr):
        if g_char in all_chars and o_char in all_chars:
            matrix.loc[g_char, o_char] += 1

# Iterate over DataFrame rows
for index, row in df_filtered_equal_length_and_distance.iterrows():
    update_confusion_matrix(row['GT'], row['OCR_text'], confusion_matrix)

# Normalize the matrix by row
confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)
confusion_matrix.fillna(0, inplace=True)
# print(confusion_matrix)


import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the heatmap
plt.figure(figsize=(15, 15))
sns.heatmap(confusion_matrix, annot=False, cmap='viridis')
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Characters')
plt.ylabel('Actual Characters')
plt.show()


# Filter out rows with all zeros
import numpy as np

# Filter out rows with all zeros
confusion_matrix_no_zeros = confusion_matrix.loc[~(confusion_matrix == 0).all(axis=1)]

# Filter out rows where the diagonal value is greater than 0.95
diagonal_values = np.diagonal(confusion_matrix_no_zeros.values)
rows_to_drop = confusion_matrix_no_zeros.index[diagonal_values > 0.8]
confusion_matrix_filtered = confusion_matrix_no_zeros.drop(rows_to_drop)

# Plotting the heatmap
plt.figure(figsize=(15, 15))
sns.heatmap(confusion_matrix_filtered, annot=False, cmap='viridis')
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Characters')
plt.ylabel('Actual Characters')
plt.show()




def apply_corrections(row, matrix, max_iterations=10):
    ocr_text = row['OCR_to_fix']
    for _ in range(max_iterations):
        corrected = ""
        changes_made = False

        for char in ocr_text:
            if char in matrix.columns:
                # Identify the most likely correct character
                likely_char = matrix[char].idxmax()
                if likely_char != char:
                    corrected += likely_char
                    changes_made = True
                else:
                    corrected += char
            else:
                corrected += char

        ocr_text = corrected

        # Break if no changes were made in this iteration
        if not changes_made:
            break

    return corrected




# Apply corrections to each row in the DataFrame
df_filtered_equal_length_and_distance['Corrected_OCR'] = df_filtered_equal_length_and_distance.apply(lambda row: apply_corrections(row, confusion_matrix_filtered), axis=1)
print(df_filtered_equal_length_and_distance.head())