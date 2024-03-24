import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap #для цветовой карты

#для цветовой карты - этими значениями разбиваем элементы на 4 подгруппы(тк трехцветный керн + пустота)
traceHold1 = 50
traceHold2 = 95
traceHold3 = 168

def custom_cmap(value):
    if value <= traceHold1:
        return 'black' 
    elif (value > traceHold1) and (value <= traceHold2):
        return 'green'
    elif (value > traceHold2) and (value <= traceHold3):
        return 'yellow'
    elif value > traceHold3:
        return 'magenta'

#работа с бин файлом
orig_x = 1240
orig_y = 320
orig_z = 320

calc_x = 10
calc_y = 10
calc_z = 2

dist_x = (orig_x - calc_x) // 2
dist_y = (orig_y - calc_y) // 2
dist_z = (orig_z - calc_z) // 2

start_x = 0
start_y = 0
start_z = 0

shift = 0

fil = open("C:/Users/nices/Downloads/Telegram Desktop/bmp_voi_.raw", 'rb')

pos = 0
pos = fil.tell()
fil.seek(pos + dist_z*(orig_x-1)*(orig_y-1) + shift)
pos = fil.tell()
fil.seek(pos + dist_y*(orig_x-1) + shift)
pos = fil.tell()
fil.seek(pos + dist_x + shift)

segment=np.zeros((calc_x - 1 - start_x, calc_y - 1 - start_y, calc_z - 1 - start_z))

max=0 #для цветовой карты, в ListedColormap понадобится наибольшее значение 

for k in range(start_x,  calc_x - 1):
    for j in range(start_y, calc_y - 1):
        for i in range(start_z, calc_z - 1):
            data = fil.read(1)
            correctData = data[0]
            
            segment[k][j][0]= correctData
            if max<correctData:
                max=correctData

        pos = fil.tell()
        fil.seek(pos + orig_x - calc_x)
    pos = fil.tell()
    fil.seek(pos + (orig_y - calc_y)*(orig_x - 1))
data_reshaped = segment.reshape((calc_x - 1 - start_x, calc_y - 1 - start_y)) #трехмерный массив 9 на 9 на 1 в двумерный 9 на 9


cmap_custom = ListedColormap([custom_cmap(i) for i in np.arange(0, max, 1)])

# Отображаем изображение с использованием своей цветовой карты
plt.imshow(data_reshaped, cmap=cmap_custom, extent=[0, calc_x - 1 - start_x, 0 ,calc_y - 1 - start_y])
plt.xticks(range(0, calc_x - start_x, 1))  # устанавливаем разметку по оси x с шагом 1
plt.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=1.0)
plt.colorbar() 
plt.title('Segment of kern')
plt.show()

