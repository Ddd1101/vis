# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from django import forms
from django.forms import fields
from django.forms import widgets
from django.core.validators import RegexValidator
from django.core.exceptions import ValidationError
from models import User,Batch
import datetime

#表单

class LoginForm(forms.Form):
    username = fields.CharField(
        required=True,
        widget=widgets.TextInput(attrs={'class': "form-control",'placeholder': '用户名'}),
        error_messages={'required': '用户名不能为空',}
    )

    pwd = fields.CharField(
        widget=widgets.PasswordInput(attrs={'class': "form-control",'placeholder': '密码'}),
        required=True,
        strip=True,
        error_messages={'required': '密码不能为空!',}
    )

    def clean(self):
        username = self.cleaned_data.get('username')
        pwd = self.cleaned_data.get('pwd')
        user = User.objects.filter(username=username).first()
        if username and pwd:
            if not user :
                self.add_error('username','用户名不存在！')
            elif pwd != user.pwd:
                self.add_error('pwd','用户名与密码不匹配！')

class RegForm(forms.Form):
    username = fields.CharField(
        required=True,
        widget=widgets.TextInput(attrs={'class': "form-control", 'placeholder': '用户名'}),
        error_messages={'required': '用户名不能为空' }
    )
    pwd = fields.CharField(
        widget=widgets.PasswordInput(attrs={'class': "form-control", 'placeholder': '密码'}),
        required=True,
        min_length=6,
        max_length=12,
        strip=True,
        error_messages={'required': '密码不能为空!'}
    )
    repwd = fields.CharField(
        widget=widgets.PasswordInput(attrs={'class': "form-control",'placeholder': '请再次输入密码!'}),
        required=True,
        min_length=6,
        max_length=12,
        strip=True,
        error_messages={'required': '请再次输入密码!'}
    )

    def clean_username(self):
        # 查找用户是否已经存在
        username = self.cleaned_data['username']
        users = User.objects.filter(username=username)
        if users:
            raise ValidationError('用户已存在')
        return username

    def _clean_new_password2(self):  # 查看两次密码是否一致
        password1 = self.cleaned_data['pwd']
        password2 = self.cleaned_data['repwd']
        if password1 and password2:
            if password1 != password2:
                self.add_error('repwd', '两次输入的密码不一致！')

    def clean(self):
        # 是基于form对象的验证，字段全部验证通过会调用clean函数进行验证
        self._clean_new_password2()  # 简单的调用而已

class UpForm(forms.Form):
    username = fields.CharField(initial='无名氏')
    date = forms.DateField(initial=datetime.date.today,disabled=True)
    dataset_name = fields.CharField(max_length=50,required=True)
    file = fields.FileField(required=True)

    def clean_datasetName(self):
        dataset_name = self.cleaned_data['dataset_name']
        datasets = Batch.objects.filter(dataset_name=dataset_name)
        if datasets:
            raise ValidationError('相同数据集名称已存在，请更改！')
        return dataset_name


