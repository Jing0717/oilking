from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import IMG, Slice

import os
import numpy as np
from PIL import Image
import cv2

import tensorflow as tf
from keras.models import load_model

# 在 MacOS 下載入過大的圖檔會跑出 OMP: Error #15:
# "Initializing libiomp5.dylib, but found libiomp5.dylib already initialized."
# 需加入下列指令, source: https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
os.environ['KMP_DUPLICATE_LIB_OK']='True'

''' change log:
------------------------------------------------------------------
[0218_2354]
  - 將上傳圖片分割, 並依對應位置輸出成 html table
[0219_1859]
  - nx/ny 用 input_number 步進器控制
  - nx/ny/img 改為 global 方便傳遞
  - 將 imgUpload.html & showImg.html 合併成分段式的 classify.html
    1.選擇上傳圖片
    2.預覽分割 (&裁切)
    3.預測
------------------------------------------------------------------
'''

graph = tf.get_default_graph()
model = load_model('./2019-02-11T1750.h5')     # 跟 manage.py 放同一層
labels = []
with open(os.path.join(os.path.dirname(__file__), "labels.txt")) as label_file:
    for line in label_file.readlines():
        labels.append(line.replace("\n",""))
img, img_slices, imgarray_slices = IMG(), Slice(), np.array([])
nx, ny = 1, 1

# Create your views here.
@csrf_exempt
def uploadImg(request):
    global img, img_slices, imgarray_slices, nx, ny

    if request.method == 'POST':
        # uploadImg 上傳圖片
        # 圖片存到資料庫
        nx = int(request.POST['a'])
        ny = int(request.POST['b'])
        if request.FILES.get('img') != None:
            img = IMG(img_origin=request.FILES.get('img'))

        img.nx = nx
        img.ny = ny
        img.slice_height = 300 / ny
        img.save()
        print(img.img_origin.path)
        print(img.img_origin.url)

        imgarray_slices, img_slices = sliceM(img.img_origin.path, img.nx, img.ny,
                                             save=True, resize=None)

        # 為了做成 html table, 重組 slices
        img_slicetable = []
        for j in range(len(img_slices[0])):
            img_slicerow = []
            for i in range(len(img_slices)):
                img_slicerow.append(img_slices[i,j])
            img_slicetable.append(img_slicerow)

        return render(request, 'classify.html', {'nx': nx, 'ny': ny,
                                                 'show_origin': True,
                                                 'img': img,
                                                 'slice_table': img_slicetable,
                                                 })
    else:
        # index 首頁
        return render(request, 'classify.html', {'nx': nx, 'ny': ny,
                                                 })
def predict(request):
    global labels
    global img, img_slices, imgarray_slices, nx, ny

    if request.method == 'POST':
        scores = classify_scores(imgarray_slices, (224, 224))
        for i in range(len(scores)):
            for j in range(len(scores[0])):
                results = []
                for k in range(len(labels)):
                    results.append(f'{labels[k]}: {scores[i,j,k]:.1%}')
                img_slices[i, j].results = results
                img_slices[i, j].result0 = f'{scores[i, j, 0]:.1%}'
                img_slices[i, j].result1 = f'{scores[i, j, 1]:.1%}'
                img_slices[i, j].result2 = f'{scores[i, j, 2]:.1%}'

        # 為了做成 html table, 重組 slices
        img_slicetable = []
        for j in range(len(img_slices[0])):
            img_slicerow = []
            for i in range(len(img_slices)):
                img_slicerow.append(img_slices[i, j])
            img_slicetable.append(img_slicerow)

        return render(request, 'classify.html', {'nx': nx, 'ny': ny,
                                                 'show_origin': True,
                                                 'img': img,
                                                 'show_prediction': True,
                                                 'slice_table': img_slicetable,
                                                 'labels': labels
                                                 })


def sliceM(filename, nx=2, ny=3, save=False, resize=None):
    # source : https://github.com/samdobson/image_slicer
    # 改寫自 main.py 裡面的 slice()
    # -----------------------------------
    # 將結果以 django model: Slice 回傳

    from math import floor
    import os
    import numpy as np
    from PIL import Image

    im = Image.open(filename)

    im_w, im_h = im.size
    columns, rows = nx, ny
    tile_w, tile_h = int(floor(im_w / columns)), int(floor(im_h / rows))

    slices = []
    img_slices = []
    loc_x = 0
    for pos_x in range(0, im_w - columns, tile_w):  # -rows for rounding error.
        slices_y = []
        img_slices_y = []
        loc_y = 0
        for pos_y in range(0, im_h - rows, tile_h):  # as above.
            area = (pos_x, pos_y, pos_x + tile_w, pos_y + tile_h)
            image = im.crop(area)
            if resize!=None:
                image = image.resize(resize)
            img_array = np.array(image)
            slices_y.append(img_array)

            base = os.path.splitext(os.path.basename(filename))[0]
            filename_new = f'{base}_{loc_x}_{loc_y}.jpg'
            image.save("./media/slices/" + filename_new, "JPEG")
            img_slices_xy = Slice(img_slice="slices/" + filename_new,
                                  pos_x=loc_x,
                                  pos_y=loc_y,
                                  img_source=filename
                                  )
            if save:
                img_slices_xy.save()
            img_slices_y.append(img_slices_xy)
            loc_y += 1

        slices.append(slices_y)
        img_slices.append(img_slices_y)
        loc_x += 1

    return np.array(slices), np.array(img_slices)


def classify_scores(img_array, resize):
    # source: https://github.com/ctudoudou/Keras-Web-demo
    # ./app/view.py
    global graph, model, labels
                                        # ex: img_array.shape = (3,4, 224,224,3)
    arr_shape = img_array.shape[0:-3]   #       arr_shape = (3,4)
    img_shape = img_array.shape[-3:0]   #       img_shape = (224,224,3)

    if len(arr_shape) == 0:
        arr0 = np.array(cv2.resize(img_array, resize)) / 255.
        arr = np.array([arr0])

    elif len(arr_shape) == 1:
        arr = []
        for i in range(arr_shape[0]):
            arr0 = arr0 = np.array(cv2.resize(img_array[i], resize)) / 255.
            arr.append(arr0)

    elif len(arr_shape) == 2:
        arr = []
        for i in range(arr_shape[0]):
            for j in range(arr_shape[1]):
                arr0 = arr0 = np.array(cv2.resize(img_array[i,j], resize)) / 255.
                arr.append(arr0)

    arr = np.array(arr)
    with graph.as_default():
        predict_scores = model.predict(arr)

    if len(arr_shape) == 2:
        predict_scores = predict_scores.reshape(list(arr_shape) + [len(labels)])

    return predict_scores
