import re
import pathlib
import requests
import json

LLM_BASE_URL = 'http://192.168.1.99:11434/api/'

MODELS = [
    'llama2:13b-chat',
    'mistral:7b-instruct',
    # 'mixtral:instruct'
]

SYSTEM_PROMPT = """Tu es un assistant juridique. Tu utilises un vocabulaire juridique. Tu m\'aides à formuler des questions\
    et problèmesjuridiques diversifiés dans tous les domaines du droit (droit civil, droit pénal, droit des affaires, droit immobilier, droit du travail,
    droit de la fonction publique, droit de l'urbanisme, droit des propriétés publiques, droit des étrangers, droit des contrats, droits des contrats administratifs, etc.)"""

class DataGeneratorPipeline:

    def __init__(self, model:str, max_loops:int=5, temperature:float=0.7, output_file:str='data/raw_synthetic_data.jsonl') -> None:
        self.model = model
        self.temperature = temperature
        self.output_file = output_file

    def run(self, prompt:str, max_loops:int=5, serialize=True):
        return self.generate_data(prompt, max_loops, serialize)

    def generate_data(self, prompt:str, max_loops:int=5, serialize:bool=True) -> list:
        generated_data = []
        i_loop = 0
        while True:
            api_endpoint = LLM_BASE_URL + 'generate'
            data = {
                'model':self.model,
                'system':SYSTEM_PROMPT,
                'prompt':prompt,
                'options':{
                    "temperature": self.temperature
                },
                "stream": False
            }
            response = requests.post(api_endpoint, json=data)
            data_ = {
                'response':json.loads(response.text)['response'],
                'model':self.model,
                'system_prompt':SYSTEM_PROMPT,
                'prompt':prompt,
                'options': data['options'],
            }
            # generated_data.append(json.loads(response.text)['response'])
            generated_data.append(data_)

            # Sauvegarder les données dans un fichier JSONL si serialize = True.
            if serialize:
                with open(self.output_file, 'a') as jsonl_file:
                    jsonl_file.write(json.dumps(data_))
                    jsonl_file.write('\n')

            i_loop += 1
            if i_loop >= max_loops:
                break

        return generated_data


    def serialize_data(self, data:list) -> None:

        first_line = None
        new_prompt = False
        if pathlib.Path(self.output_file).exists():
            with open(self.output_file, 'r') as jsonl_file:
                first_line = jsonl_file.readlines()[0].strip()

        if first_line != self.prompt.strip():
            print('New prompt')
            new_prompt = True

        with open(self.output_file, 'a+') as jsonl_file:
            if new_prompt:
                jsonl_file.write(self.prompt + '\n')
            for element in data:
                jsonl_file.write(json.dumps(element))
                jsonl_file.write('\n')


if __name__ == '__main__':

    model = 'mistral:7b-instruct'
    prompts = [
        'Génère le maximum de questions avec un problème juridique en francais.',
        'Génère le maximum de questions posant un problème juridique en francais.',
        'Génère le maximum de questions soulevant un problème juridique en francais.',
        'Génère le maximum de questions avec un problème juridique concret en francais.',
        'Génère le maximum de questions posant un problème juridique concret en francais.',
        'Génère le maximum de questions soulevant un problème juridique concret en francais.',
        'Génère le maximum de problèmes juridiques en francais.',
        'Génère le maximum de problèmes juridiques concrets en francais.',
    ]
    for model in MODELS:
        for prompt in prompts:
            prompt = 'Génère le maximum de questions avec un problème juridique en francais.'
            DataGeneratorPipeline(model, temperature=0.9).run(prompt, max_loops=10)
