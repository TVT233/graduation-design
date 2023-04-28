import os
PATH = r'C:\Users\13291\Downloads\yolov5-master\images\test'
names = os.listdir(PATH)
for name in names:
    name = name.split('.')
    if len(name[0]) == 2:
        oldname = name[0]
        name[0] = str(int(name[0])+100)
        if len(name[0]) == 2:
            front = '000'
        else:
            front = '00'
        os.rename(PATH + '/' + oldname + '.' + name[1], PATH + '/' + front + name[0] + '.' + name[1])
