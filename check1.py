file = open(r'C:\Users\13291\Downloads\yolov5-master\process\result\40\length_width_info\1_1.txt', 'r')
lines = file.readlines()
for i in lines:
    print(i.strip('\n'))
