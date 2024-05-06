import re
from django.utils.timezone import datetime
from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

def home(request):
    return HttpResponse("Image to be processed...")

def hello_there(request, name):
    # print(request.build_absolute_uri())
    now = datetime.now()
    # formatted_now = now.strftime("%A %d %B, %Y at %X")
    # return HttpResponse(name + formatted_now)
    return render(
        request,
        'hello/hello_there.html',
        {
            'name': name,
            'date': datetime.now()
        }
    )