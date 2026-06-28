from django.urls import include, path

from classifier import views

urlpatterns = [
    path("health", views.health, name="health"),
    path("ready", views.ready, name="ready"),
    path("api/v1/", include("classifier.urls")),
]
