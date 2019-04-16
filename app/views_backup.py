from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import IMG, Slice
# from image.settings import STATICFILES_DIRS
# from image.settings import MEDIA_ROOT

import os
import numpy as np
from PIL import Image
import cv2

import tensorflow as tf
import keras
from keras.models import load_model

# 在 MacOS 下載入過大的圖檔會跑出 OMP: Error #15:
# "Initializing libiomp5.dylib, but found libiomp5.dylib already initialized."
# 需加入下列指令, source: https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
os.environ['KMP_DUPLICATE_LIB_OK']='True'

graph = tf.get_default_graph()
model = load_model('./2019-02-11T1750.h5')     # 跟 manage.py 放同一層
labels = []
with open(os.path.join(os.path.dirname(__file__), "labels.txt")) as label_file:
    for line in label_file.readlines():
        labels.append(line.replace("\n",""))

# Create your views here.
@csrf_exempt
def uploadImg(request):
    global labels

    if request.method == 'POST':
        # 上傳圖片之後存下來的路徑
        # img_file=os.getcwd().replace('\\','/')+'/media/upload'
        # 切割圖片之後存檔的路徑
        # slicer_img_path='C:/Users/user/Desktop/django_learn/image/media/crop'

        # 刪除上傳過的圖及切割過的圖
        # clearfile(img_file)
        # clearfile(slicer_img_path)

        # 取得imgUpload.html內 name=img的檔案"名稱""
        # img_file = request.FILES.get('img')
        # print(img_file)

        # 圖片存到資料庫
        img = IMG(img_origin=request.FILES.get('img'),
                  nx=int(request.POST['a']),
                  ny=int(request.POST['b'])
                  )
        # print(type(img))
        # 並將此圖片存在model.py內的"IMG function"內upload_to的資料夾內
        img.save()

        # print(STATICFILES_DIRS)
        # file=str(STATICFILES_DIRS[0])+str(img_file)
        # print(file)
        print(img.img_origin.path)
        print(img.img_origin.url)

        # nx = int(request.POST['a'])
        # ny = int(request.POST['b'])

        # slicer(file, nx, ny)
        # slicer(img.img_field.path, nx, ny)
        # dir_slices = os.path.join(MEDIA_ROOT, "slices")
        # slices, filename_slices = slicexy(img.img_origin.path, img.nx, img.ny, save=True, save_to=dir_slices)
        imgarray_slices, img_slices = sliceM(img.img_origin.path, img.nx, img.ny, save=True, resize=(224,224))

        # s = filename_slices.flatten()
        # print(s)

        scores = classify_scores(imgarray_slices, (224,224))
        print(scores.shape)
        print(scores)

        # img_slices = []
        results = []
        for i in range(len(scores)):
            # img_slices_x = []
            results_x = []
            for j in range(len(scores[0])):
                results_xy = []
                for k in range(len(labels)):
                    results_xy.append(f'{labels[k]}: {scores[i,j,k]:.1%}')
                # img_slices_xy = Slice(img_slice=filename_slices[i,j],
                #                       pos_x=i,
                #                       pos_y=j,
                #                       result=results_xy,
                #                       img_source=img.img_origin.path,
                #                       )
                results_x.append(results_xy)
                img_slices[i,j].result = results_xy
                # img_slices_x.append(img_slices_xy)
            results.append(results_x)
            # img_slices.append(img_slices_x)

        print(results)
        results_flat = np.array(results).reshape(len(results) * len(results[0]), len(labels))
        img_slices_flat = img_slices.flatten()



        # return render(request, 'imgupload.html')
        # return render(request, 'showImg.html')
        # return showImg(request)
        return render(request, 'showImg.html', {'img': img,
                                                # 'imgs': s,
                                                'slices': img_slices_flat,
                                                'results': results_flat,
                                                })

    else:
        print('開啟第一頁')
        # img = IMG(img_field=request.FILES.get('img'))
        # img.save()
        return render(request, 'imgupload.html')


# def showImg(request):
#     imgs = IMG.objects.all()
#     context = {
#         'imgs' : imgs
#     }
#     return render(request, 'showImg.html', context)


# def showImg(request):
#     # IMG.delete()
#     imgs = IMG.objects.all()
#     for s in imgs:
#         print(s.img_field)
#
#     # 切割圖片之後存檔的路徑
#     slicer_path = 'C:/Users/user/Desktop/django_learn/image/media/crop'
#     files = os.listdir(slicer_path)
#     s = []
#     for file in files:
#         # print(file)
#         # if os.path.isfile(slicer_path+'/'+file) is True:
#         a = 'crop' + '/' + file
#         s.append(a)
#         # print(s)
#
#     # img = IMG(img_field=request.FILES.get('img'))
#     # context = {
#     #     'imgs' : img
#     # }
#     # return render(request, 'showImg.html', {'imgs':imgs})
#     return render(request, 'showImg.html', {'imgs': s})


# def delete(request):
#     print("要刪除資料庫內的數據")
#     # img = IMG(img_field=request.FILES.get('img'))
#     imgs = IMG.objects.all()
#     imgs.delete()
#     # return render_to_response("imgupload.html", {})
#     return uploadImg(request)


# def slicer(file, nx, ny):
#     photo = cv2.imread(file)
#     # print(photo.shape)
#     heigh, weigh, color = photo.shape
#     print("要切割圖的規格:" + "高:" + str(heigh), "寬:" + str(weigh), "顏色:" + str(color))
#     h = 0
#     w = 0
#
#     heigh = int(heigh)
#     weigh = int(weigh)
#
#     cut_heigh = int(heigh / int(ny))
#     cut_weigh = int(weigh / int(nx))
#     cut_float_h = heigh % int(ny)
#     cut_float_w = weigh % int(nx)
#     a = 0
#     for h_position in range(h, heigh, cut_heigh):
#         if h_position + cut_float_h != heigh:
#             b = 0
#             for w_position in range(w, weigh, cut_weigh):
#                 #     print("寬起始點",w_position)
#                 #     print("高起始點",h_position)
#                 if w_position + cut_float_w != weigh:
#                     crop = photo[h_position:h_position + cut_heigh, w_position:w_position + cut_weigh]
#                     # 將切割好的圖片存檔至指定路徑
#                     cv2.imwrite('C:/Users/user/Desktop/django_learn/image/media/crop/crop' + str(a) + str(b) + '.jpg',
#                                 crop)
#                     # print("檔名:crop"+str(a)+str(b)+".jpg")
#                 b += 1
#         a += 1


# def clearfile(path):
#     # print(path)
#     # print(type(os.listdir(path)))
#     allfiles = os.listdir(path)
#     # print(allfiles)
#     for file in allfiles:
#         print("刪掉切圖的檔名:" + file)
#         if os.path.isfile(path + '/' + file) is True:
#             os.remove(path + '/' + file)


def slicexy(filename, nx=2, ny=3, save=False, save_to=None):
    # source : https://github.com/samdobson/image_slicer
    # 改寫自 main.py 裡面的 slice()
    from math import floor
    import os
    import numpy as np
    from PIL import Image

    im = Image.open(filename)

    im_w, im_h = im.size
    columns, rows = nx, ny
    tile_w, tile_h = int(floor(im_w / columns)), int(floor(im_h / rows))

    slices = []
    for pos_x in range(0, im_w - columns, tile_w):  # -rows for rounding error.
        slices_y = []
        for pos_y in range(0, im_h - rows, tile_h):  # as above.
            area = (pos_x, pos_y, pos_x + tile_w, pos_y + tile_h)
            image = im.crop(area)
            img_array = np.array(image)
            slices_y.append(img_array)
        slices.append(slices_y)

    slices = np.array(slices)
    if save:
        base = os.path.splitext(os.path.basename(filename))[0]
        if save_to != None:
            dirname = save_to
        else:
            dirname = os.path.dirname(filename)

        filenames=[]
        for x in range(0, nx):
            filenames_y = []
            for y in range(0, ny):
                filename_new = f'{base}_{x}_{y}.jpg'
                filepath_new = os.path.join(dirname, filename_new)
                # print(filename_new)
                Image.fromarray(slices[x, y]).save(filepath_new)
                filenames_y.append(filename_new)
            filenames.append(filenames_y)

        return np.array(slices), np.array(filenames)

    else:
        return np.array(slices)

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
            print(img_slices_xy.img_slice)
            img_slices_y.append(img_slices_xy)
            loc_y += 1

        slices.append(slices_y)
        img_slices.append(img_slices_y)
        loc_x += 1

    return np.array(slices), np.array(img_slices)


def classify_scores(img_array, resize):
    # source: https://github.com/ctudoudou/Keras-Web-demo
    # ./app/view.py
    global graph
    global model
    global labels
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