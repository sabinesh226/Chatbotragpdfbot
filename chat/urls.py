from django.contrib import admin
from django.urls import path
from chat.views import chat_home, get_response

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', chat_home),
    path('get-response/', get_response),
]