from django.shortcuts import render


def train_model_view(request):
    return render(request, 'train_model.html')
