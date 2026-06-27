import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lazycat.llm import prism_client

logging.basicConfig(level=logging.DEBUG)

async def main():
    print("Testing Prism Agent Call with Tools...")
    prism_client.url = "http://10.0.0.16:7777"
    
    mock_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
    
    try:
        response = await prism_client.call_agent(
            model="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
            messages=[{"role": "user", "content": "What is the weather in Seattle?"}],
            system_prompt="You are a helpful assistant.",
            agent_name="CUSTOM_TEST_AGENT",
            tools=[mock_tool],
            stream=False,
        )
        print(f"Success! Status: {response.status_code}")
        print(response.json())
    except Exception as e:
        print(f"Error calling Prism: {e}")
        if hasattr(e, "response") and e.response:
            print("Response Body:")
            print(e.response.text)

if __name__ == "__main__":
    asyncio.run(main())
