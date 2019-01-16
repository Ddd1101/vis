from django.conf.urls import url
from App import views,ajaxView
from django.views.generic import TemplateView

urlpatterns=[
    url(r'^$',views.history),
    url(r'^display/$', views.display),
    url(r'^usergroup/$',views.usergroup),
    url(r'^history/$',views.history),
    url(r'^upload/$', TemplateView.as_view(template_name="upload.html"),name='upload'),
    url(r'^table/$',views.table,name='table'),
    url(r'^login/$',views.login,name='login'),
    url(r'^register/$',views.register,name='register'),
    url(r'^logout/$',views.logout,name='logout'),
    url(r'^ajax-batch-data/$', ajaxView.doBatchData, name='ajax-batch-data'),
    url(r'^ajax-batches-list/$',ajaxView.getBatchesNames,name='ajax-batches-list'),
    url(r'^ajax-batch-result/$',ajaxView.getResult,name='ajax-batch-result'),
    url(r'^ajax-mistake-analysis/$',ajaxView.getMistakeRate,name='ajax-mistake-analysis'),
    url(r'^ajax-forge-analysis/$', ajaxView.getForgeRate, name='ajax-forge-analysis'),
    url(r'^ajax-range-analysis/$',ajaxView.getRangeRate,name='ajax-range-analysis'),
    url(r'^ajax-user-graph/$',ajaxView.getPersonaData,name='ajax-user-graph'),
    url(r'^ajax-uploadFile/$', ajaxView.uploadFile, name='uploadFile'),
    url(r'^start-train/$',ajaxView.buildModel,name='start-train'),
    url(r'^ajax-check/$',ajaxView.doCheck,name='ajax-check'),
    url(r'^ajax-custom-config/$',ajaxView.editCustomConfig,name='ajax-custom-config')
]