import os


def create_folder(PATH):

    try:
        if not os.path.exists(PATH):
            os.makedirs(PATH)
    except OSError:
        print('Error: Creating directory' + PATH)
