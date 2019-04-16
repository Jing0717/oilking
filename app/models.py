from django.db import models

# Create your models here.
class IMG(models.Model):
    # upload_to為圖片上傳的路徑，不存在就創建一個新的。
    img_origin = models.ImageField(upload_to='upload', null=True)       # /media/upload
    nx = models.IntegerField(null=True)
    ny = models.IntegerField(null=True)
    slice_height = models.IntegerField(null=True)

class Slice(models.Model):
    img_slice = models.ImageField(upload_to='slices', null=True)        # /media/slices
    # img_slice = models.CharField(null=True, max_length=100)
    pos_x = models.IntegerField(null=True)
    pos_y = models.IntegerField(null=True)
    result_text = models.CharField(null=True, max_length=500)
    img_source = models.CharField(null=True, max_length=100)

    bg_color = models.CharField(null=True, max_length=100)

    result0 = models.FloatField(null=True) #lebal0的機率
    result1 = models.FloatField(null=True)
    result2 = models.FloatField(null=True)

    result0txt = models.CharField(null=True, max_length=100)
    result1txt = models.CharField(null=True, max_length=100)
    result2txt = models.CharField(null=True, max_length=100)
