from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('predictor/', views.predict_view, name='predict_view'),
    path('batch/', views.batch_view, name='batch_view'),
    path('analytics/', views.analytics_view, name='analytics_view'),
    path('contact/', views.contact_view, name='contact_view'),
    path('predict/', views.predict, name='predict'),
    path('batch-predict/', views.batch_predict, name='batch_predict'),
    path('contact-submit/', views.contact, name='contact'),
]
