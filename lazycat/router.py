import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from lazycat.llm import PrismClient

logger = logging.getLogger(__name__)

class IntentDefinition(BaseModel):
    name: str
    description: str

class IntentRouter:
    """
    Generic intent classifier that routes user queries to the appropriate capability
    based on predefined intent definitions.
    """
    
    def __init__(self, intents: List[IntentDefinition], llm_client: Optional[PrismClient] = None):
        self.intents = intents
        self.llm_client = llm_client or PrismClient()
        
    def _build_system_prompt(self) -> str:
        intent_list = "\n".join([f"{i+1}. {intent.name}: {intent.description}" for i, intent in enumerate(self.intents)])
        intent_names = [f'"{intent.name}"' for intent in self.intents]
        intent_names_str = " | ".join(intent_names)
        
        return f"""You are a routing agent for an autonomous system.
Your task is to classify the user's query into one of the following intents:
{intent_list}

Output ONLY a valid JSON object with this schema:
{{
  "intent": {intent_names_str},
  "query": string,
  "reasoning": string,
  "extracted_entities": dict
}}
Do not include any extra chat or markdown formatting outside the JSON block.
"""

    async def classify(
        self, 
        user_input: str, 
        conversation_history: List[Dict[str, str]] = None,
        model: str = "gpt-4o",
        provider: str = "vllm"
    ) -> Dict[str, Any]:
        """
        Classifies the user input to determine the core intent.
        """
        messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ]
        
        if conversation_history:
            messages.extend(conversation_history[-4:])
            
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = await self.llm_client.call_agent(
                model=model,
                provider=provider,
                messages=messages,
                system_prompt=messages[0]["content"],
                agent_name="generic_router",
                max_tokens=256
            )
            
            # The response.text might contain markdown code blocks (e.g., ```json\n...\n```)
            content = response.text.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            return json.loads(content.strip())
        except Exception as e:
            logger.error(f"Failed to classify intent: {e}")
            return {
                "intent": self.intents[-1].name if self.intents else "UNKNOWN",
                "query": user_input,
                "reasoning": f"Fallback due to failure: {str(e)}",
                "extracted_entities": {}
            }
