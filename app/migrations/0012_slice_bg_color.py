# Generated by Django 2.1.5 on 2019-02-20 04:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0011_auto_20190219_1622'),
    ]

    operations = [
        migrations.AddField(
            model_name='slice',
            name='bg_color',
            field=models.CharField(max_length=100, null=True),
        ),
    ]
