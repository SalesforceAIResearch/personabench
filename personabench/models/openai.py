import openai

class OpenAIGPT:
    def __init__(self, api_key, model) -> None:
        self.api_key = api_key
        self.model = model
        # openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def generate(self, prompt):
        if "system" in prompt:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt['system']},
                    {"role": "user", "content": prompt['instruction']}
                ]
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt['instruction']}
                ]
            )
        return response.choices[0].message.content.strip()