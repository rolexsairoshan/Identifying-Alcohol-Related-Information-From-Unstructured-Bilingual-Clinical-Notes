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
    address= models.CharField(max_length=3000)
    gender= models.CharField(max_length=30)

class identify_alcohol_information(models.Model):

    Fid= models.CharField(max_length=3000)
    pid= models.CharField(max_length=3000)
    gender= models.CharField(max_length=3000)
    age= models.CharField(max_length=3000)
    hypertension= models.CharField(max_length=3000)
    heart_disease= models.CharField(max_length=3000)
    clinical_note= models.CharField(max_length=3000)
    ever_married= models.CharField(max_length=3000)
    work_type= models.CharField(max_length=3000)
    Residence_type= models.CharField(max_length=3000)
    avg_glucose_level= models.CharField(max_length=3000)
    bmi= models.CharField(max_length=3000)
    smoking_status= models.CharField(max_length=3000)
    stroke= models.CharField(max_length=3000)
    Prediction= models.CharField(max_length=3000)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



