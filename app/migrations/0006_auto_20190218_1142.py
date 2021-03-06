# Generated by Django 2.1.7 on 2019-02-18 11:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0005_auto_20190218_0547'),
    ]

    operations = [
        migrations.CreateModel(
            name='Slice',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('img_slice', models.ImageField(null=True, upload_to='crop')),
                ('pos_x', models.IntegerField(null=True)),
                ('pos_y', models.IntegerField(null=True)),
                ('result', models.CharField(max_length=500, null=True)),
                ('img_source', models.CharField(max_length=100, null=True)),
            ],
        ),
        migrations.DeleteModel(
            name='IMG_Slices',
        ),
    ]
