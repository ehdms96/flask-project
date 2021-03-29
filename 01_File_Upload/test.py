import os

def fileList():
    file_list = os.listdir("./uploads")
    file_list_png = [file for file in file_list if file.endswith(".png")]
    html = "file_list_png: {}".format(file_list_png)
    return html
