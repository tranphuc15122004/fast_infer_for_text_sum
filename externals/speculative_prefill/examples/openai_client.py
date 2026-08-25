from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="local_server",
)

completion = client.chat.completions.create(
  model="meta-llama/Meta-Llama-3.1-8B-Instruct",
  messages=[
    {"role": "user", "content": "Hello! Can you tell me a bit more about who is Magnus Carlsen? "}
  ]
)

print(completion.choices[0].message)
