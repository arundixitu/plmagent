import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv(
    "OPENAI_MODEL",
    "gpt-6-luna"
)


if not api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found in the .env file."
    )


client = OpenAI(
    api_key=api_key
)


print("\nAUREON OPENAI API CONNECTION TEST")
print("=" * 80)

print("Model:", model)


response = client.responses.create(
    model=model,
    input=(
        "Reply with exactly this sentence: "
        "Aureon AI connection successful."
    )
)


print("\nMODEL RESPONSE")
print("-" * 80)

print(response.output_text)


print("\nAPI TEST COMPLETE.")