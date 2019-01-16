# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# Create your views here.
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from forms import LoginForm, RegForm, UpForm
from models import User,Batch,Batch_record
import re
from util import getConfig


# 主页,可视化页
def display(req):
    config = getConfig()
    return render(req,'display.html',{'config':config})

def usergroup(req):
    return render(req,'users.html')

def history(req):
    return render(req,'history.html')

#表格数据查看页
def table(req):
    datasets = list(Batch.objects.values('dataset_name').distinct())
    return render(req,'table.html',{'datasets':datasets})


#注册
def register(req):
    if req.method == 'POST':
        uf = RegForm(req.POST)
        if uf.is_valid():
            #获得表单数据
            username = uf.cleaned_data['username']
            pwd=uf.cleaned_data['pwd']
            #添加到数据库
            user=User.objects.create(username= username,pwd=pwd)
            req.session['user']=user.username
            return render(req,'register.html',{'uf':uf,'success':True})
    else:
        uf = RegForm()
    return render(req,'register.html',{'uf':uf},)

#登陆
def login(req):
    if req.method == 'POST':
        uf = LoginForm(req.POST)
        if uf.is_valid():
            username = uf.cleaned_data['username']
            user = User.objects.get(username=username)
            req.session['user'] = user.username
            return HttpResponseRedirect("/")
        else:
            return render(req, 'login.html', {'uf': uf})
    else:
        uf = LoginForm()
    return render(req,'login.html',{'uf':uf})

#退出
def logout(req):
    try:
        del req.session['user']
    except KeyError:
        pass
    response = HttpResponseRedirect('/login')
    #清理cookie
    response.delete_cookie('user')
    return response