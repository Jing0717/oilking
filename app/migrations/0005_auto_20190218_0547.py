# Generated by Django 2.1.7 on 2019-02-18 05:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0004_auto_20190218_0534'),
    ]

    operations = [
        migrations.AddField(
            model_name='img_slices',
            name='img_source',
            field=models.CharField(max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='img_slices',
            name='pos_x',
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name='img_slices',
            name='pos_y',
            field=models.IntegerField(null=True),
        ),
    ]
