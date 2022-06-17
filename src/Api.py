
# IMPORTS
# import spacy
# import random
from fastapi import FastAPI
import json
from typing import List
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import AnalyzerEngine,EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import SpacyNlpEngine, NlpArtifacts, NlpEngineProvider
from presidio_anonymizer.entities.engine import OperatorConfig
from faker import Faker

################### GLOBALS ##################
SPACY_MODEL_PATH = "models/custom_spacy_models/400docs"

################### API STUFF ####################
description = """
This script is responsible for detecting certain types of entities in the input text document, which are about contracts. 
Once the entities are recognized, they are processed according to their type.
Some of them will be replaced by <ENTITYTYPE>, encrypted, masked with a certain character, etc.
"""
tags_metadata = [{
        "name": "anonimizar_documento",
        "description": "Anonymize a spanish text. Main pipeline. It will return the annonimized text as well as the annotations.",
    }]

hcommonk_anonymizer = FastAPI(
    title = "Hcommonk-Anonymizer",
    description = description,
    openapi_tags = tags_metadata
)
################### RECOGNIZERS ##################

class Contracts_recognizer(EntityRecognizer):
    """
    Class that inherits from EntityRecognizer and is responsible of recognizing certain 
    custom entities of the text (those that correspond to contracts).
    """
    expected_confidence_level = 0.7 # expected confidence level for this recognizer
    def load(self) -> None:
        """No loading is required."""
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts) -> List[RecognizerResult]:
        """
        Logic for detecting a specific PII.
        We are going to use the entities atribute since 
        the spacy model that we are using has the capabilities to recognise those entities.
        """
        results = []
        for ent in nlp_artifacts.entities:
            results.append(
                RecognizerResult(
                    entity_type=ent.label_,
                    start=nlp_artifacts.tokens[ent.start].idx,
                    end=nlp_artifacts.tokens[ent.start].idx + len(ent.text),
                    score=self.expected_confidence_level
                )
            )
        return results

def build_operators(actions_json, associations):
    """
    Function that builds a certain data structure containing instructions on how to operate 
    with each type of entity. Some of these operations are : masking, substitution, encryption, hashing, etc.
    """
    fake = Faker('es_ES')
    
    operators = {}
    for key,value in actions_json.items():
        if value[0].lower() == 'mask':
            operators[key] = OperatorConfig(operator_name="mask", params={'masking_char': value[1], 'chars_to_mask': value[2], 'from_end': False})
        elif value[0].lower() == 'replace':
            operators[key] = OperatorConfig(operator_name="replace", params={'new_value': value[1]})
        elif value[0].lower() == 'encrypt':
            operators[key] = OperatorConfig(operator_name="encrypt", params={'key': value[1]})
        elif value[0].lower() == 'hash':
            operators[key] = OperatorConfig(operator_name="encrypt", params={'hash_type': 'md5'})
            #hash_types = ['sha256', 'sha512', 'md5']# what should we use? default sha256
        elif value[0].lower() == 'random replace':
            # figure out which fake random entity to replace
            if 'city' in key.lower():
                new_value = fake.city()
            elif 'name' in key.lower():
                new_value = fake.name()
            else:
                print(f'Error entity type not supported for replacement ({key}).')
                exit()
            operators[key] = OperatorConfig(operator_name="replace", params={'new_value': new_value})
        elif value[0].lower() == 'none':
            operators[key] = OperatorConfig("custom", {"lambda": lambda x: x})
        else:
            print(f'Operation {value} not supported. Check the action file for the entity type {key}')
            operators[key] = OperatorConfig("custom", {"lambda": lambda x: x})

    # Add default operator. If no operation then we leave the entity as it is.
    operators['DEFAULT'] = operators[key] = OperatorConfig("custom", {"lambda": lambda x: x})

    # Add the associations (for example the entity ES_NIF equals FounderIDNumber)
    for key,value in associations.items():
        if key in operators:
            operators[value] = operators[key]
    
    return operators

def create_es_anonymizer():
    """ Function to create the anonymizer. """
    anonymizer = AnonymizerEngine()
    return anonymizer
    
def create_es_analyzer(all_entities):
    """ 
    Function to create the analyzer. First load the custom spacy model, 
    then create the Analyzer engine and finally add the custom recognizers to the registry of the Analyzer.
    """
    # Create configuration containing engine name and models
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "es", "model_name": SPACY_MODEL_PATH}],
    }    
    # Create NLP engine based on configuration
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    # the languages are needed to load country-specific recognizers 
    # for finding phones, passport numbers, etc.
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine,supported_languages=["es"])
    contracts_recognizer = Contracts_recognizer(supported_entities=all_entities, supported_language= 'es')
    analyzer.registry.add_recognizer(contracts_recognizer)
    #print(analyzer.get_supported_entities(language='es'))
    return analyzer

################### INIT ################
# 0. Prepare all the supported entities. Including associations of previous (default) supported entites with the custom ones.
all_entities = ['FounderName','FounderContribution','BusinessName','FounderIDNumber','FounderAddress','FounderCityName','AdminName','AdminType','OtherContribution','CompanyAddress']
associations = {
    'FounderIDNumber' : 'ES_NIF'
}
all_entities.extend(list(associations.values()))
ALL_ENTITIES = list(set(all_entities))

# 1. Build custom operators (load default actions file per entity type)
with open('src/actions.json') as json_file:
    actions = json.load(json_file)
CUSTOM_OPERATORS = build_operators(actions_json=actions['Contratos'], associations = associations)

# 2. Input data (examples or default data)

# 3. Create custom spanish analyzer and anonymizer. Anonymizer is built with custom recognizer (spacy retrained model)
ES_ANALYZER = create_es_analyzer(all_entities)
ES_ANONYMIZER= create_es_anonymizer()
    
#response = anonimizar_documento(text, es_analyzer, es_anonymizer, operators = custom_operators, wanted_entities = all_entities)

################### API F ################
@hcommonk_anonymizer.post("/anonimizar_documento", tags=["anonimizar_documento"])
def anonimizar_documento(text: str, actions = actions, wanted_entities = all_entities):
    """
    Anonymize a spanish text. Main pipeline. It will return the annonimized text as well as the annotations.
    """
    #custom_operators = build_operators(actions_json=actions['Contratos'], associations = associations)
    annotations = ES_ANALYZER.analyze(text=text, language='es', entities=ALL_ENTITIES)
    anonymized_text = ES_ANONYMIZER.anonymize(text=text, analyzer_results=annotations, operators=CUSTOM_OPERATORS).text
    response= {
        'text': anonymized_text,
        'annotations': annotations
        }
    return response

#uvicorn api:hcommonk_anonymizer --reload
