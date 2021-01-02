import os
import pandas as pd

def read_files(category):
    """
    Input: State a category to pick all the messages from
    Output: A dataset with two columns: text of the message, and a tag of the category associated to the message
    """
    files_content_list = []
    category_path = os.path.join('dataset', category)
    files_names_list = os.listdir(category_path)
    for file_name in files_names_list:
        file_path = os.path.join(category_path, file_name)
        content = open(file_path, 'r', errors='ignore').read()
        files_content_list.append(content)
        df_files_content = pd.DataFrame(files_content_list, columns=['Text'])
        df_files_content['Category'] = category
    return df_files_content
