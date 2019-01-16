# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt,csrf_protect
from models import Batch, Batch_record,User
from forms import UpForm
from dataProcess import SeqModel,TypeModel,RangeModel
from scipy.stats import gaussian_kde
import util
from util import str2bool,delKeys
import json,re
import numpy as np


# 根据batch_id返回原始数据到table
def doBatchData(req):
    if req.method == 'GET':
        batch_id = req.GET['batch_id']
        queryset = Batch_record.objects.filter(batch_id__id=batch_id).values('serial_no','capacity','loss_angle_tangent','leakage_current')
        dict={
            'total':len(queryset),
            'rows':tuple(queryset)
        }
        return JsonResponse(dict,safe=False)


# 根据数据文件名获取所有数据批次meta信息
def getBatchesNames(req):
    dataset_name = req.POST['dataset_name']
    data = list(Batch.objects.filter(dataset_name=dataset_name).values('model','specification','batch_no','id'))
    return JsonResponse(data,safe=False)


# 返回各批次数据，是前端boostrapTable请求的函数
def getResult(req):
    # TODO 后端分页
    batches = list(Batch.objects.values())
    rows=[]
    config=util.getConfig()
    for batch in batches:
        alert=[]
        if batch['mistake_rate']>config['mistake']['threshold']:
            alert.append('mistake_rate')
        if batch['forge_rate']>config['forge']['threshold']:
            alert.append('forge_rate')
        if batch['range_rate']>config['range']['threshold']:
            alert.append('range_rate')
        row = {
            "id":batch['id'],
            "model":batch['model'],
            "specification":batch['specification'],
            "batch_no":batch['batch_no'],
            "forge_rate":batch['forge_rate'],
            "mistake_rate":batch['mistake_rate'],
            "range_rate":batch['range_rate'],
            "record_date":batch['record_date'],
            "product":batch['dataset_name'],
            "alert":alert
        }
        rows.append(row)
    # 根据forge_rate降序返回
    def rowcmp(a,b):
        if a['forge_rate']>b['forge_rate']:
            return -1
        elif a['forge_rate']<b['forge_rate']:
            return 1
        else:
            return 0
    sorted_rows=sorted(rows,cmp=rowcmp)

    return JsonResponse(sorted_rows,safe=False)

# 根据ID返回误输率分析结果，可视化分析
def getMistakeRate(req):
    batch_id = req.POST['batch_id']
    product_name = Batch.objects.get(id=batch_id).dataset_name
    typeModel = TypeModel(name=product_name)
    batch_data = tuple(Batch_record.objects.filter(batch_id__id=batch_id).values_list('capacity', 'loss_angle_tangent', 'leakage_current'))
    pred_score,trans_sample,normal_sample=typeModel.getGraphData(batch_data)
    Batch.objects.filter(id=batch_id).update(mistake_rate=pred_score)
    kernel = gaussian_kde(trans_sample)
    samples_y = kernel.pdf(trans_sample)
    kernel = gaussian_kde(normal_sample)
    normal_y = kernel.pdf(normal_sample)
    obj={
        'mistake_rate':pred_score,
        'samples_x':trans_sample,
        'samples_y': samples_y.tolist(),
        'normal_x':normal_sample,
        'normal_y':normal_y.tolist()
    }
    return JsonResponse(obj,safe=False)

def getForgeRate(req):
    batch_id = req.POST['batch_id']
    batch_data = tuple(Batch_record.objects.filter(batch_id__id=batch_id).values_list('capacity', 'loss_angle_tangent', 'leakage_current'))
    product_name = Batch.objects.get(id=batch_id).dataset_name
    sqModel = SeqModel(name=product_name)
    forge_rate,seg_num,matrix = sqModel.getGraphData(batch_data)
    Batch.objects.filter(id=batch_id).update(forge_rate=forge_rate)
    obj = {
        'forge_rate':forge_rate,
        'seg_num':seg_num,
        'score_matrix':matrix
    }
    return JsonResponse(obj,safe=False)

def getRangeRate(req):
    batch_id = req.POST['batch_id']
    batch_data = tuple(Batch_record.objects.filter(batch_id__id=batch_id).values_list('capacity', 'loss_angle_tangent','leakage_current'))
    product_name = Batch.objects.get(id=batch_id).dataset_name
    rgModel = RangeModel(name=product_name)
    batch_data = np.asarray(batch_data)
    range_rate,data,normal_data = rgModel.getGraphData(batch_data)
    Batch.objects.filter(id=batch_id).update(range_rate=range_rate)
    obj={
        'range_rate':range_rate,
        'data':data,
        'normal_data':normal_data
    }
    return JsonResponse(obj,safe=False)

# 数据检查,行为预测
def doCheck(req):
    id_list = req.POST['id_list']
    id_list = json.loads(id_list)
    for id in id_list:
        batch = Batch_record.objects.filter(batch_id__id=id)
        batch_data=tuple(batch.values_list('capacity', 'loss_angle_tangent', 'leakage_current'))
        product_name=Batch.objects.get(id=id).dataset_name
        sqModel=SeqModel(name=product_name)
        forge_rate = sqModel.predict([batch_data])[0]
        typeModel = TypeModel(name=product_name)
        mistake_rate = typeModel.predict([batch_data])[0]
        rgModel = RangeModel(name=product_name)
        batch_data = np.asarray(batch_data)
        range_rate = rgModel.predict([batch_data])[0]
        Batch.objects.filter(id=id).update(mistake_rate=mistake_rate,forge_rate=forge_rate,range_rate=range_rate)
    return JsonResponse({
        'success':True
    },safe=False)

def getPersonaData(req):
    # todo
    import numpy as np
    from sklearn.cluster import KMeans
    users = []
    num=200
    for i in range(num):
        arr = np.random.normal(size=3)
        u = np.tanh(abs(arr/3))
        users.append(u.tolist())
    model = KMeans(n_clusters=8)
    labels = model.fit_predict(users)
    return JsonResponse({
        'users':users,
        'labels':labels.tolist()
    },safe=False)
    pass

# 修改custom参数设置，并且根据参数重新检测或者修改alert结果
@csrf_exempt
def editCustomConfig(req):
    key = req.POST['key']
    if key == 'forge':
        util.customConfig(int(req.POST['gap_length']),'forge','gap_length')
        util.customConfig(float(req.POST['threshold']),'forge','threshold')
    elif key =='mistake':
        util.customConfig(float(req.POST['threshold']), 'mistake', 'threshold')
    elif key == 'range':
        util.customConfig(float(req.POST['threshold']),'range','threshold')
    return JsonResponse({
        'success':True
    },safe=False)

def uploadFile(req):
    if req.method == 'POST':
        my_form = UpForm(req.POST, req.FILES)
        if my_form.is_valid():
            #f = my_form.cleaned_data['file']
            try:
                handle_uploaded_file(req.FILES['file'].read(),my_form.cleaned_data)
                return JsonResponse({'success':True})
            except ValueError,arg:
                return JsonResponse({'success':False,'ErrorMsg':arg.message})
        else:
            return JsonResponse({'success':False})

    return JsonResponse({'success':False})


def handle_uploaded_file(f,formDict):
    import xlrd
    datasets = xlrd.open_workbook(filename=None,file_contents=f)
    sheets = datasets.sheets()
    try:
        for i in range(len(sheets)):
            # 创建Batch
            sheet = sheets[i]
            meta = sheet.row_values(0)
            try:
                patt = ur'型号[^a-zA-Z0-9]*(\w+)' if isinstance(meta[0],unicode) else r'型号[^a-zA-Z0-9]*(\w+)'
                reObj = re.match(patt,meta[0])
                model = reObj.group(1)
            except:
                model = ''
            try:
                patt = ur'规格[^a-zA-Z0-9]*(\S+)' if isinstance(meta[1],unicode) else r'规格[^a-zA-Z0-9]*(\S+)'
                reObj = re.match(patt,meta[1])
                specification = reObj.group(1)
            except:
                specification = ''
            try:
                patt = ur'批号[^a-zA-Z0-9]*(\w+)' if isinstance(meta[4],unicode) else r'批号[^a-zA-Z0-9]*(\w+)'
                reObj = re.match(patt,meta[4])
                batch_no = reObj.group(1)
            except:
                batch_no=''
            batch=Batch.objects.create(
                dataset_name=formDict['dataset_name'],
                model = model,
                specification = specification,
                batch_no = batch_no,
                recorder_id=User.objects.get(username=formDict['username']).id,
                record_date=formDict['date']
            )
            data =[]
            for j in range(3,sheet.nrows):
                row = sheet.row_values(j)
                batch_record=Batch_record.objects.create(
                    batch=batch,
                    serial_no = row[0],
                    capacity = row[1],
                    loss_angle_tangent = row[2],
                    leakage_current = row[3]
                )
                data.append(row[1:4])
    except:
        raise ValueError,'上传文件格式不符合要求'


def buildModel(req):
    batches = Batch.objects.all()
    samples = []
    for batch in batches:
        batch_data = Batch_record.objects.filter(batch=batch)
        samples.append(batch_data)
    typeModel = TypeModel(name='tantalum')
    typeModel.train(samples)
    seqModel = SeqModel(name='tantalum')
    seqModel.train(samples)

    return JsonResponse({})