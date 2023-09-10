from django.db import models
from djongo import models as djongo_models


class TrainedModels(models.Model):
    _id = djongo_models.ObjectIdField(db_column='_id', primary_key=True)
    model_name = models.TextField(db_column='model_name', default='')
    path = models.TextField(db_column='path', default='')
    accuracy = models.FloatField(db_column='accuracy', default=0.0)
    loss = models.FloatField(db_column='loss', default=0.0)
    created_at = models.DateTimeField(db_column='created_at', auto_now_add=True)
    upload_service_type = models.TextField(db_column='service_type', default='')
    is_active = models.BooleanField(db_column='is_active', default=True)

    @classmethod
    def create_from_dict(cls, data: dict):
        """
        This method is used to create a TrainedModels object from a dictionary.

        Parameters
        ----------
        data: dict
            This is the dictionary that contains the data to create the object.

        Returns
        -------
        TrainedModels: The created object.
        """
        return cls(**data)