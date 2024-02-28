# Ce module contient les fonctions pour prétraiter les données synthétiques générées par des LLMS.

import pathlib
import re
import csv, json
from typing import List
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


RAW_DATA = f'{pathlib.Path().home()}/projects/public_private_law_classifier/data/raw_questions.txt'

# def extract_split_chatGPT_questions() -> List[str]:
#     with open(RAW_DATA, 'r') as file:
#         raw_data = file.read()

#     RGX_QUESTIONS = re.compile(r'(\n)(.*?\?)')

#     matches = RGX_QUESTIONS.findall(raw_data)
#     matches = [match[1].strip() for match in matches]
#     matches = set(matches)
#     for match in matches:
#         print(match)

#     print(len(matches))

#     with open('dataset.csv', 'w', newline='') as csvfile:
#         csv_writer = csv.writer(
#             csvfile,
#             delimiter=';',
#             quotechar='|',
#             quoting=csv.QUOTE_MINIMAL
#         )
#         csv_writer.writerow(('question', 'source', 'label'))
#         for match in matches:
#             csv_writer.writerow((match, 'ChatGPT_3.5', ''))

#     return matches

# # Définir une fonction par défaut pour extraire les questions.
# def extract_split_default_questions(filepath:str) -> List[str]:

#     # Lire le fichier.
#     with open(filepath, 'r') as file:
#         raw_data = file.readlines()

#     # Ne pas lire la première ligne. Supprimer les lignes vides.
#     raw_data = [
#         json.loads(line).strip()
#         for line in raw_data[1:]
#         if len(json.loads(line).strip())
#     ]

#     # Extraire les questions en utilisant une expression régulière.
#     RGX_QUESTIONS = re.compile(r'(\n[\s\d\.]*)(.*?\?)')
#     matches = []
#     for line in raw_data:
#         rgx_matches = RGX_QUESTIONS.findall(line)
#         temp_matches = [rgx_match[1].strip() for rgx_match in rgx_matches]
#         temp_matches = list(set(temp_matches)) # Supprimer les doublons
#         matches.extend(temp_matches)

#     return matches


# Définir une fonction par défaut pour extraire les questions.
def extract_split_default_questions(filepath:str) -> List[str]:

    # Lire le fichier.
    with open(filepath, 'r') as file:
        raw_data = file.readlines()

    # Extraire les questions en utilisant une expression régulière.
    RGX_QUESTIONS = re.compile(r'(\n[\s\d\.]*)(.*?\?)')
    matches = []
    for line in raw_data:

        # Ne pas traiter les lignes vides.
        if not len(line.strip()):
            continue

        rgx_matches = RGX_QUESTIONS.findall(json.loads(line).get('response'))
        temp_matches = [rgx_match[1].strip() for rgx_match in rgx_matches]
        temp_matches = list(set(temp_matches)) # Supprimer les doublons
        matches.extend(temp_matches)

    return matches


# Définir une fonction pour extraire les questions générées depuis l'UI de ChatGPT.s
def extract_split_chatGPT_questions() -> List[str]:
    with open(RAW_DATA, 'r') as file:
        raw_data = file.read()

    RGX_QUESTIONS = re.compile(r'(\n)(.*?\?)')

    matches = RGX_QUESTIONS.findall(raw_data)
    matches = [match[1].strip() for match in matches]
    matches = set(matches)

    with open('dataset.json', 'w', newline='') as jsonfile:
        rows = []
        for match in matches:
            rows.append({
                'question': match,
                'source': 'ChatGPT_3.5',
                'label': ''
            })

        jsonfile.write(json.dumps(rows))

    return matches


# Définir une fonction de stemmatisation utilisable avec pandas.s
def stemmatize(text):

    stemmer = SnowballStemmer(language='french')

    #perform stemming using PorterStemmer for all non-english-stopwords
    stemmatized_words = [
        stemmer.stem(words)
        for words in word_tokenize(text, language="french")
        if words not in set(stopwords.words("french"))
    ]

    return stemmatized_words



if __name__ == '__main__':
    # extract_split_chatGPT_questions()
    # with open('data/generated_data_mistral:7b-instruct_t0.7.jsonl', 'r') as jsonl_file:
    #     data = jsonl_file.readlines()
    #     print(data, len(data))

    filepath = 'data/raw_synthetic_data.jsonl'
    matches = extract_split_default_questions(filepath)
    print(matches, len(matches))
