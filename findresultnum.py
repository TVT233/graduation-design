import os

result_path = r'C:/Users/13291/Downloads/yolov5-master/process/result'

def findrn(path):
    list = os.listdir(path)
    if len(list) == 0:
        rn = 1
        return rn
    else:
        max = 0
        for i in list:
            if int(i) > max:
                max = int(i)
        rn = max + 1
        return rn

if __name__ == '__main__':
    print(findrn(result_path))