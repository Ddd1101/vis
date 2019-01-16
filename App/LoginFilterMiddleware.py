# -*- coding: utf-8 -*-
from django.conf import settings
from re import compile
from django.http import HttpResponseRedirect

try:
    from django.utils.deprecation import MiddlewareMixin
except ImportError:
    MiddlewareMixin = object

EXEMPT_URLS=['/login/','/register/']

class LoginRequiredMiddleware(MiddlewareMixin):

    def process_request(self,request):
        if request.is_ajax():
            # TODO token验证
            pass
        else:
            if not request.session.has_key('user') and  request.path not in EXEMPT_URLS:
                return HttpResponseRedirect('/login/')
