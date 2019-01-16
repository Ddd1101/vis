# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from django.db import models

# Create your models here.

# 解析并存储上传的excel表格

class User(models.Model):
    username = models.CharField(max_length=50)
    pwd = models.CharField(max_length=50)

    def __unicode__(self):
        return self.username


class Batch(models.Model):

    model = models.CharField(max_length=10)
    specification = models.CharField(max_length=20)
    batch_no = models.CharField(max_length=20)
    record_date = models.DateField()
    mistake_rate = models.FloatField(null=True,blank=True)
    range_rate = models.FloatField(null=True,blank=True)
    forge_rate = models.FloatField(null=True,blank=True)
    dataset_name = models.CharField(max_length=50)
    recorder = models.ForeignKey(User)

    def __unicode__(self):
        return self.batch_no

class Batch_record(models.Model):
    batch = models.ForeignKey(Batch)
    serial_no = models.IntegerField()
    capacity = models.FloatField()
    loss_angle_tangent = models.FloatField()
    leakage_current = models.FloatField()

    def __unicode__(self):
        return u'%s %s' % (self.batch, self.serial_no)

    def to_dict(self):
        dict={}
        dict['serial_no']=self.serial_no
        dict['capacity']=self.capacity
        dict['loss_angle_tangent']=self.loss_angle_tangent
        dict['leakage_current']=self.leakage_current
        return dict

