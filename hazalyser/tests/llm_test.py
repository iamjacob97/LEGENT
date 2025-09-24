from openai import OpenAI
from hazalyser.helpers import get_env_key

deepseek = get_env_key("DSV3")
client = OpenAI(base_url=deepseek["base_url"], api_key=deepseek["api_key"])

response = client.chat.completions.create(
  model=deepseek["model"],
  messages=[
    {
      "role": "user",
      "content": [{"type":"text", "text":"What is the meaning of life?"}]
    }
  ],
  temperature=0.2,
  max_completion_tokens=2048
)
output = response.choices[0].message.content
usage = response.usage
print(usage.prompt_tokens)