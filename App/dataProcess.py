# -*- coding:utf-8 -*-
#
from __future__ import division
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import time
import pickle
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from math import exp,pi,sqrt,tanh
from sklearn.covariance import EllipticEnvelope
from sklearn.externals import joblib
from django.conf import settings
from util import getConfig


def gaussian(point, ndim, sigma):
    x = point[0]
    y = point[1]
    center_x = center_y = (ndim - 1) / 2
    midRes = exp(-((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)) / (2 * sigma * sigma))
    result = midRes / (2 * pi * sigma * sigma)
    return result

def normalize(data,type='zscore'):
        if type == 'zscore':
            # z-score标准化
            normalized = preprocessing.scale(data)
        elif type == 'minmax':
            # 最大最小标准化
            scaler = MinMaxScaler()
            scaler.fit(data)
            normalized = scaler.transform(data)
        else:
            raise ValueError
        return normalized

# 计算数值输入的舒适度,digit间的距离之和除以digits数量
# 数值越高，舒适度越低，输入一行记录，得出这一行记录的舒适度得分
# @method 表示不同输入方式，'single'单指，'blind'四指盲打
# @mode 'full'表示计算整个数值舒适度，'part'表示计算小数点及之后的
def comfort_score(num,method='single',mode='full'):
    score = 0
    coords = {
        '0': [0.5, 0], '.': [2, 0],
        '1': [0, 1], '2': [1, 1], '3': [2, 1],
        '4': [0, 2], '5': [1, 2], '6': [2, 2],
        '7': [0, 3], '8': [1, 3], '9': [2, 3]
    }
    num = str(num)
    if method == 'single':
        # 考虑单指输入
        if mode=='part':
            location = num.find('.')
            num = num[location:]
        for i in range(1,len(num)):
            digit = num[i]
            former = num[i-1]
            dist = sqrt((coords[digit][0]-coords[former][0])*(coords[digit][0]-coords[former][0])+\
                   (coords[digit][1]-coords[former][1])*(coords[digit][1]-coords[former][1]))
            score += dist
        score/=(len(num))
    elif method == 'blind':
        # 考虑四指盲打 TODO 角度计算分数？
        pass
    return score

def outlierDetection(data,proportion):
    clf = EllipticEnvelope(support_fraction=1.,contamination=proportion)
    data = data.reshape((-1, 1))
    clf.fit(data)
    labels = clf.predict(data)
    return labels

def smith_waterman(a, b, alignment_score=3, gap_cost=1):
    H = np.zeros((len(a) + 1, len(b) + 1))
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            match = H[i - 1, j - 1] + (alignment_score if np.array_equal(a[i - 1], b[j - 1]) else -gap_cost)
            delete = H[i - 1, j] - gap_cost
            insert = H[i, j - 1] - gap_cost
            H[i, j] = max(match, delete, insert, 0)
    return H

def sequence_piece_align(data,seg_length=150,gap_length=10,gap=True):
    feature=[0,0]
    l = len(data)
    stats=[]
    matrixs=[]
    seg_num = int(l / seg_length)
    if seg_num < 4:
        seg_num = 4
    for r in range(seg_num):
        lo = int(r/seg_num*l)
        hi = int((r+1)/seg_num*l)
        cur_data = data[lo:hi]
        if gap:
            ll = int(len(cur_data) / gap_length)
            gaps = np.array_split(cur_data, ll)
            first = [gaps[i] for i in range(ll) if i % 2 == 0]
            second = [gaps[i] for i in range(ll) if i % 2 != 0]
            a = np.concatenate(first).tolist()
            b = np.concatenate(second).tolist()
        else:
            ll = int(len(cur_data)/2)
            a = cur_data[:ll]
            b=cur_data[ll:]

        H = smith_waterman(a,b)
        stats.append(H.max())
        matrixs.append(H.tolist())
    while 0 in stats:
        stats.remove(0)
    if len(stats) >0:
        mean = np.mean(stats)
        nrange = np.max(stats)-np.min(stats)
        feature = [nrange/mean,np.max(stats)/(l/seg_num/2)]

    return feature,matrixs




class SeqModel:

    def __init__(self,name,threshold=0.7,seg_length=150):
        self.threshold=threshold
        self.name=name
        self.seg_length=seg_length

    # 计算gap为true和false两种情况下的序列内部相似度差异，用这个作为序列特征
    # @mode multiple表示转换多个数据集，single转换一个
    def transform(self,dataset,mode='multiple'):
        X=[]
        if mode == 'multiple':
            for data in dataset:
                X.append(sequence_piece_align(data, gap=True, seg_length=self.seg_length)[0])
                X.append(sequence_piece_align(data, gap=False, seg_length=self.seg_length)[0])
            return np.asarray(X)
        else:
            X.append(sequence_piece_align(dataset,gap=True, seg_length=self.seg_length)[0])
            X.append(sequence_piece_align(dataset,gap=False, seg_length=self.seg_length)[0])
            return np.asarray(X)

    def train(self,samples):
        X=self.transform(samples,mode='multiple')
        model = Birch(threshold=0.05)
        model.fit(X)
        joblib.dump(model,settings.MODEL_DIR+'seq_'+self.name+'.pkl')


    def getNormalModel(self):
        try:
            model = joblib.load(settings.MODEL_DIR+'seq_'+self.name+'.pkl')
        except:
            raise ValueError('please train model first')
        # 找最大簇
        labels = model.labels_
        max_size = 0
        max_cluster_label = labels[0]
        for l in set(labels.tolist()):
            size = len(labels[labels == l])
            if size > max_size:
                max_size = size
                max_cluster_label = l
        # 最大簇质心作为簇心

        X = model.subcluster_centers_
        max_cluster = X[model.subcluster_labels_ == max_cluster_label]
        self.normal_model = np.mean(max_cluster, axis=0)
        return self.normal_model


    def predict(self, samples):
        probs=[]
        for sample in samples:
            points = self.transform(sample, mode='single')
            c = self.getNormalModel()
            prob = 0
            for point in points:
                dist = sqrt((c[0] - point[0]) * (c[0] - point[0]) + (c[1] - point[1]) * (c[1] - point[1]))
                if tanh(dist) > prob:
                    prob = tanh(dist)
            probs.append(prob)
        return probs

    def getGraphData(self,sample):
        seg_num = int(len(sample)/self.seg_length)
        X1=sequence_piece_align(sample, gap=True, seg_length=self.seg_length)
        X2=sequence_piece_align(sample, gap=False, seg_length=self.seg_length)
        c = self.getNormalModel()
        probs = []
        for point in [X1[0],X2[0]]:
            dist = sqrt((c[0] - point[0]) * (c[0] - point[0]) + (c[1] - point[1]) * (c[1] - point[1]))
            probs.append(tanh(dist))
        if probs[0]>probs[1]:
            return probs[0],seg_num,X1[1]
        else:
            return probs[1],seg_num,X2[1]


# 结合序列权值字典给出每个记录的输入得分
# 数值越高，舒适度越低，输入一行记录，得出这一行记录的舒适度得分
# @mode 'full'表示计算整个数值舒适度，'part'表示计算小数点及之后的
# TODO 调整
def getTypeScore(num,mode='full'):
    score = 0
    coords = {
        '0': [0.5, 0], '.': [2, 0],
        '1': [0, 1], '2': [1, 1], '3': [2, 1],
        '4': [0, 2], '5': [1, 2], '6': [2, 2],
        '7': [0, 3], '8': [1, 3], '9': [2, 3]
    }
    num = str(num)
    # 考虑单指输入
    if mode=='part':
        location = num.find('.')
        num = num[location:]
    for i in range(1,len(num)):
        digit = num[i]
        former = num[i-1]
        dist = sqrt((coords[digit][0]-coords[former][0])*(coords[digit][0]-coords[former][0])+\
               (coords[digit][1]-coords[former][1])*(coords[digit][1]-coords[former][1]))
        score += dist
    score/=(len(num))
    return score

class TypeModel:

    def __init__(self,name,tol=1e-3,threshold=0.6):
        self.clusters={}
        self.models={}
        self.name=name
        self.tol=tol
        self.threshold=threshold

    # 将数据值转换为输入序列得分，用数据的序列得分代替原数据
    def transform(self,dataset):
        X = []
        for data in dataset:
            data = np.asarray(data)
            shape = data.shape
            type_scores = np.asarray([getTypeScore(l, mode='full') for l in data.flatten()])
            type_scores = type_scores.reshape(shape)
            line_scores = np.asarray([np.sum(l) for l in type_scores])
            X.append(line_scores)
        return np.asarray(X)

    def trainCluster(self,samples):
        num_set = []
        for data in samples:
            values = np.asarray(data)
            s1 = set(values[:, 0].tolist())
            s2 = set(values[:, 1].tolist())
            s3 = set(values[:, 2].tolist())
            num_set.append([len(s1), len(s2), tanh(len(s3))])
        num_set = normalize(num_set, 'minmax')
        model = Birch()
        labels = model.fit_predict(num_set)
        joblib.dump(model,settings.MODEL_DIR+'type_dist_'+self.name+'.pkl')
        return labels

    def predictCluster(self,samples):
        num_set = []
        for data in samples:
            values = np.asarray(data)
            s1 = set(values[:, 0].tolist())
            s2 = set(values[:, 1].tolist())
            s3 = set(values[:, 2].tolist())
            num_set.append([len(s1), len(s2), tanh(len(s3))])
        num_set = normalize(num_set, 'minmax')
        model = joblib.load(settings.MODEL_DIR+'type_dist_'+self.name+'.pkl')
        labels = model.predict(num_set)
        return labels

    def train(self,samples):
        X = self.transform(samples)
        labels = self.trainCluster(samples)
        for i in range(len(samples)):
            if self.clusters.has_key(labels[i]):
                plot = normalize(np.reshape(X[i], (-1, 1)), 'minmax')
                self.clusters[labels[i]].extend(plot)
            else:
                plot = normalize(np.reshape(X[i], (-1, 1)), 'minmax')
                self.clusters[labels[i]]=[]
                self.clusters[labels[i]].extend(plot)
        for label, data in self.clusters.iteritems():
            score = 0
            for n_components in range(1,6):
                model = GaussianMixture(n_components=n_components)
                model.fit(data)
                if exp(model.score(data))-score < self.tol:
                    model = GaussianMixture(n_components=n_components-1)
                    model.fit(data)
                    break
                else:
                    score = exp(model.score(data))
            self.models[label] = model
            joblib.dump(model,settings.MODEL_DIR+'type_'+self.name+'_'+str(label)+'.pkl')

    def predict(self,samples):
        X = self.transform(samples)
        labels = self.predictCluster(samples)
        pred_score=[]
        cnt = 0
        for label,sample in zip(labels,X):
            cnt+=1
            model = joblib.load(settings.MODEL_DIR+'type_'+self.name+'_'+str(label)+'.pkl')
            sample = normalize(np.reshape(sample,(-1,1)),'minmax')
            pred_score.append(1-tanh(exp(model.score(sample))))
        return pred_score

    def getGraphData(self,sample):
        X = self.transform([sample])
        label = self.predictCluster([sample])[0]
        model = joblib.load(settings.MODEL_DIR+'type_'+self.name+'_'+str(label)+'.pkl')
        trans_sample = normalize(np.reshape(X[0],(-1,1)),'minmax')
        pred_score = 1-tanh(exp(model.score(trans_sample)))
        trans_sample = trans_sample.squeeze()
        model_sample = model.sample(len(sample))[0].squeeze()
        return pred_score,trans_sample.tolist(),model_sample.tolist()


class RangeModel:
    def __init__(self,name,threshold=0.05,win_size=20,step_size=20):
        self.name = name
        self.threshold=threshold
        self.win_size=win_size
        self.step_size=step_size
        with open(settings.MODEL_DIR+'range_model.pkl') as file:
            self.groups = pickle.load(file)
        
    def floatingWindow(self,samples):
        win_size = self.win_size
        step_size = self.step_size
        floats = []
        labels = []
        for data in samples:
            l = len(data)
            plot = []
            data = data[:, 0]
            if max(data) < 2:
                group = 0
            elif max(data) >= 2 and max(data) < 20:
                group = 1
            else:
                group = 2
            start = 0
            while start + win_size < l:
                seg = data[start:start + win_size]
                start += step_size
                # cv是差异系数 表示波动状态
                cv = np.std(seg, axis=0) / np.mean(seg, axis=0)
                plot.append(cv)
            floats.append(plot)
            labels.append(group)
        return floats, labels

    def train(self,samples):
        samples,labels=self.floatingWindow(samples)
        group = {0:[],1:[],2:[]}
        for i in range(len(samples)):
            label=labels[i]
            sample = samples[i]
            group[label].extend(sample)
        self.groups = group

    def predict(self,samples):
        samples, labels = self.floatingWindow(samples)
        probs=[]
        for i in range(len(samples)):
            label = labels[i]
            sample = samples[i]
            if len(sample)==0:
                continue
            miu = np.mean(self.groups[label])
            sigma = np.std(self.groups[label])
            cnt = 0
            for p in sample:
                if p<miu-3*sigma or p>miu+3*sigma:
                    cnt+=1
            probs.append(cnt/len(sample))
        return probs

    def getGraphData(self,sample):
        score = self.predict([sample])[0]
        samples, labels = self.floatingWindow([sample])
        sample = samples[0]
        label = labels[0]
        num = len(self.groups[label])
        theta1 = np.arange(num) / num * 2 * np.pi
        data1 = self.groups[label]
        num = len(sample)
        theta = np.arange(num)/num*2*np.pi
        data = np.concatenate((np.reshape(sample,(-1,1)),np.reshape(theta,(-1,1))),axis=1)
        normal_data = np.concatenate((np.reshape(data1,(-1,1)),np.reshape(theta1,(-1,1))),axis=1)
        return score,data.tolist(),normal_data.tolist()