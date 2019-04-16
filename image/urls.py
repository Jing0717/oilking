"""image URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.9/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""

from django.conf.urls import url
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from app import views

# urlpatterns = [
#     url(r'^admin/', admin.site.urls),
# ]


urlpatterns = [
    url('admin/', admin.site.urls),         # admin
    url(r'^$', views.uploadImg),            # 首頁 = 上傳圖片的地方
    url('uploadImg/', views.uploadImg),     # 上傳圖片的地方
    url('predict/', views.predict),         # 丟入 model 並預測
    ] +\
    static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

