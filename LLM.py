import ollama

class EvaluteNeeds:
    def getNeeds(self, transcript: str, context: str):
        response = ollama.chat(
            model='llama3',
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": transcript}
            ]
        )
        return response['message']['content']