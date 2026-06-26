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

    @staticmethod
    async def self_review_html(html_content: str, llm_client, project: str = "lazycat-sdk-app") -> str:
        """
        Agentic loop to review generated HTML for dead buttons or missing handlers.
        """
        system_prompt = (
            "You are an expert UI Reviewer. Your task is to verify that the provided HTML component "
            "is fully functional and interactive.\n\n"
            "RULES:\n"
            "1. If you find any `onclick=\"return false\"` or dead buttons/links without actual JS handlers, YOU MUST FIX THEM.\n"
            "2. Ensure any interactive elements (tabs, filters, sorts, modals) have inline `<script>` tags that implement their logic.\n"
            "3. Ensure IDs are namespaced (e.g., `id=\"my-unique-table-1\"`).\n"
            "4. Return ONLY the final corrected raw HTML string. Do not use markdown wrappers like ```html."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Review and fix this HTML if needed:\n\n{html_content}"}
        ]
        
        try:
            resp = await llm_client.call_agent(
                model="gpt-4o",
                messages=messages,
                system_prompt=system_prompt,
                agent_name="html_reviewer",
                max_tokens=4096,
                temperature=0.1,
                project=project
            )
            content = resp.text.strip()
            if content.startswith("```html"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            return content.strip()
        except Exception as e:
            logger.error(f"Failed self-review loop: {e}")
            return html_content
