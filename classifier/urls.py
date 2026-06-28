from django.urls import path

from classifier.views import ClassifyView

urlpatterns = [
    path("classify", ClassifyView.as_view(), name="classify"),
]
