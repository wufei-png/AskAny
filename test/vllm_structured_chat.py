"""Test vLLM chat with structured output support."""

# https://docs.vllm.ai/en/latest/features/structured_outputs/#online-serving-openai-api
import sys
from pathlib import Path

# Add project root to path to enable imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openai import OpenAI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from askany.config import settings  # noqa: E402
from askany.main import initialize_llm  # noqa: E402


class Info(BaseModel):
    name: str
    age: int


def test_vllm_chat_with_structured_output():
    """Test vLLM chat completion with JSON schema structured output."""
    # Initialize LLM using initialize_llm() from askany/main.py:266
    llm, embed_model = initialize_llm()

    # Get configuration from settings (same as initialize_llm uses)
    api_base = settings.openai_api_base
    api_key = settings.openai_api_key if settings.openai_api_key else None
    model = settings.openai_model

    # Get underlying LLM for display purposes
    if hasattr(llm, "_llm"):  # AutoRetryVLLM wrapper
        underlying_llm = llm._llm
    else:  # Direct OpenAI LLM
        underlying_llm = llm

    # Create OpenAI client directly from configuration
    # For vLLM, api_key can be None or empty string, but OpenAI client requires it
    # Use empty string as fallback for vLLM (vLLM typically doesn't require auth)
    client_api_key = api_key if api_key else ""
    client = OpenAI(
        api_key=client_api_key,
        base_url=api_base,
    )

    print(f"Using LLM: {type(llm)}")
    print(f"Underlying LLM: {type(underlying_llm)}")
    print(f"API Base: {api_base}")
    print(f"Model: {model}")
    print("-" * 80)

    # Create chat completion with structured output
    # Reference: dev_readme/vllm.md - response_format supports json_schema
    completion = client.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "我是吴非，今年22岁",
            }
        ],
        # response_format={
        #     "type": "json_schema",
        #     "json_schema": {
        #         "name": "car-description",
        #         "schema": json_schema,
        #     },
        # },
        response_format=Info,
    )

    # Print the response
    response_content = completion.choices[0].message
    print("Response:")
    assert response_content.parsed
    print("Name:", response_content.parsed.name)
    print("Age:", response_content.parsed.age)
    print("-" * 80)

    # Try to parse the response as JSON and validate against the model
    # try:
    #     import json

    #     response_json = json.loads(response_content)
    #     car = CarDescription(**response_json)
    #     print("Parsed and validated successfully:")
    #     print(f"  Brand: {car.brand}")
    #     print(f"  Model: {car.model}")
    #     print(f"  Type: {car.car_type}")
    # except Exception as e:
    #     print(f"Failed to parse/validate response: {e}")

    return completion


if __name__ == "__main__":
    test_vllm_chat_with_structured_output()
