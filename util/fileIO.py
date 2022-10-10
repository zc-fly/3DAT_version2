import os


def videoFileFinder(path):
    """
    find all video file in specify path, return bas path
    """
    target = ['.mp4', '.avi', '.rmvb', '.mov']

    checkList = []
    all_file = os.listdir(path)
    for each in all_file:
        if not os.path.isdir(each):
            extension = os.path.splitext(each)[1]
            if extension in target:
                checkList.append(path +"/" +each)
            else:
                pass
    return checkList