from django.http import JsonResponse
from rest_framework.views import APIView
from django.core import serializers

from .models import AllPredictions


# Create your views here.

class AllPredictionsView(APIView):
    @staticmethod
    def get(request):
        get_data = request.GET
        limit = int(get_data.get('limit', 10))
        offset = int(get_data.get('offset', 0))

        all_predictions = AllPredictions.objects.all()[offset:offset + limit]
        all_predictions_serialized = serializers.serialize('json', all_predictions)

        print(AllPredictions.objects.all().values('prediction_id'))

        def map_predictions(prediction):
            return {
                'pk': prediction.pk,
                'image': prediction['fields']['image'],
                'prediction': prediction['fields']['prediction'],
                'confidence': prediction['fields']['confidence']
            }

        return JsonResponse(all_predictions_serialized, safe=False, status=200)
