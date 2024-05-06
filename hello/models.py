from django.db import models
from django.utils import timezone
# Create your models here.

class LogMessage(models.Model):
    message = models.CharField(max_length=300)
    log_date = models.DateTimeField("date logged")

    # a model class can include methods that return values computed from other class properties.
    # Models typically include such a method that returns a string representation of the instance
    def __str__(self) -> str:
        date = timezone.localtime(self.log_date)
        return f"'{self.message}' logged on {date.strftime('%A, %d %B, %Y at %X')}"
    