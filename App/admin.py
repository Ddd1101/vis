# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib import admin
from models import User,Batch,Batch_record

# Register your models here.
admin.site.register([User,Batch,Batch_record])