from openai import OpenAI

client = OpenAI(
    api_key="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",
    base_url="https://api.groq.com/openai/v1"
)

models = client.models.list()
print(models)