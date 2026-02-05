import os
import logging
import hashlib
from tenacity import (retry, stop_after_attempt,  # for exponential backoff
                      wait_random_exponential)

from openai import OpenAI


# Initialize OpenAI client lazily to avoid errors when not using OpenAI features
CLIENT = None

def get_client():
    """Get or create OpenAI client. Only initializes when actually needed."""
    global CLIENT
    if CLIENT is None:
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "This is only required when using GPT models (--metric=llm_gpt-* or --entailment_model=gpt-*)."
            )
        CLIENT = OpenAI(api_key=api_key)
    return CLIENT


@retry(wait=wait_random_exponential(min=1, max=10))
def predict(prompt, temperature=1.0, model='gpt-4'):
    """Predict with GPT-4 model."""

    if isinstance(prompt, str):
        messages = [
            {"role": "user", "content": prompt},
        ]
    else:
        messages = prompt

    if model == 'gpt-4':
        model = 'gpt-4-turbo'  # or 'gpt-4o'
    elif model == 'gpt-3.5':
        model = 'gpt-3.5-turbo'

    client = get_client()  # Get client only when actually needed
    output = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=200,
        temperature=temperature,
    )
    response = output.choices[0].message.content
    return response


def md5hash(string):
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)
