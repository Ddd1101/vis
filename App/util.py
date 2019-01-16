# -*- coding: utf-8 -*-
import json
import collections
from django.conf import settings

def getConfig():
    with open(settings.CONF_DIR+'Default.json','r') as file:
        default = json.load(file)
    with open(settings.CONF_DIR+'User.json','r') as file:
        user = json.load(file)
    config = dict(default)
    config=update(config,user)

    return config

def customConfig(value,*keys):
    with open(settings.CONF_DIR+'User.json','r') as file:
        old = json.load(file)
        nkeys=len(keys)
        custom = {keys[-1]:value}
        for i in range(nkeys-1):
            key = keys[nkeys-i-2]
            custom = {key:custom}
        config = dict(old)
        config=update(config,custom)
    with open(settings.CONF_DIR + 'User.json', 'w') as file:
        json.dump(config,file)


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

# 用u更新d，嵌套递归更新
def update(d, u):
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            r = update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d

# 删除dict d中所有键为key的entry
def delKeys(d,key):
    c = d.copy()
    for k, v in d.iteritems():
        if k == key:
            del c[k]
        elif isinstance(v,collections.Mapping):
            r = delKeys(c[k],key)
            c[k] =r
    return c

if __name__ =='__main__':
    a ={'a':{'kd':222},'k':{'dd':3,'kd':2}}
    c=delKeys(a,'kd')
    print c