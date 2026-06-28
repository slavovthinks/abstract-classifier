from django.conf import settings
from rest_framework import serializers

from arxiv_ml.enums import Category


class ClassifyRequestSerializer(serializers.Serializer):
    """Validate the inbound classification request.

    ``abstract`` is required, non-empty, and length-capped to reject absurdly
    long input before it ever reaches the model. The cap is sourced from
    ``MAX_ABSTRACT_CHARS`` so it stays a single configurable knob.
    """

    abstract = serializers.CharField(
        required=True,
        allow_blank=False,
        trim_whitespace=True,
        max_length=settings.INFERENCE_CONFIG.max_abstract_chars,
        error_messages={
            "required": "The 'abstract' field is required and must be non-empty.",
            "blank": "The 'abstract' field is required and must be non-empty.",
            "max_length": "The 'abstract' field exceeds the maximum allowed length.",
        },
    )


class CategoryScoreSerializer(serializers.Serializer):
    category = serializers.ChoiceField(choices=[c.value for c in Category])
    confidence = serializers.FloatField()


class PredictionResponseSerializer(serializers.Serializer):
    predicted_category = serializers.ChoiceField(choices=[c.value for c in Category])
    confidence = serializers.FloatField()
    top_k = CategoryScoreSerializer(many=True)
    model_version = serializers.CharField()
