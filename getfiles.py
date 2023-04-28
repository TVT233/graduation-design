import os

FileRoot = r'C:/Users/13291/Downloads/yolov5-master/process/buffer'

def getfiles(FileRoot):
    path_list = []
    name_list = []
    filenames = os.listdir(FileRoot)
    for filename in filenames:
        a = os.path.join(FileRoot, filename)
        path_list.append(a)
        name_list.append(os.path.splitext(filename)[0])
    return path_list, name_list

if __name__ == '__main__':
    f, n = getfiles(FileRoot)
    print(f)
    print(n)

