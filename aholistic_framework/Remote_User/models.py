from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)

class GIS_DataSets_model(models.Model):

    city_name=models.CharField(max_length=300)
    names=models.CharField(max_length=300)
    area_name=models.CharField(max_length=300)
    mobile_name=models.CharField(max_length=300)
    app_name=models.CharField(max_length=300)
    wd_name=models.CharField(max_length=300)
    Affected_person_name=models.CharField(max_length=300)
    safe_status=models.CharField(max_length=300)
    crime_desc=models.CharField(max_length=300)
    crime_date=models.CharField(max_length=300)
    action_taken=models.CharField(max_length=300)
    crime_prevention=models.CharField(max_length=300)
    No_Of_Case=models.CharField(max_length=300)


class GIS_DataSets_Trained_model(models.Model):

    city_name = models.CharField(max_length=300)
    names = models.CharField(max_length=300)
    area_name = models.CharField(max_length=300)
    mobile_name = models.CharField(max_length=300)
    app_name = models.CharField(max_length=300)
    wd_name = models.CharField(max_length=300)
    Affected_person_name = models.CharField(max_length=300)
    safe_status = models.CharField(max_length=300)
    crime_desc = models.CharField(max_length=300)
    crime_date = models.CharField(max_length=300)
    action_taken = models.CharField(max_length=300)
    crime_prevention = models.CharField(max_length=300)
    No_Of_Case = models.CharField(max_length=300)
    ctype= models.CharField(max_length=300)

class ctype_ratio_model(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)


class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)
