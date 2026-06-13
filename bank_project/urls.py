from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('batch-predict/', views.batch_predict, name='batch_predict'),
]
