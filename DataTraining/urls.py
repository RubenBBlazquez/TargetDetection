"""HackingToolsWebCore URL Configuration """

from django.urls import path
from django.conf.urls.static import static

from Core import settings
from .views import train_model_view

urlpatterns = [
    path('train/', train_model_view, name="trainCVModel")
]
