import logging
from typing import Any, Dict
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

class Validator:
    @staticmethod
    def validate_schema(model_class: type[BaseModel], data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates a dictionary against a Pydantic BaseModel.
        Returns the validated dictionary or raises ValueError.
        """
        try:
            instance = model_class(**data)
            return instance.model_dump()
        except ValidationError as e:
            logger.error(f"Schema validation failed for {model_class.__name__}: {e}")
            raise ValueError(f"Validation failed: {e}")
