"""
This script is responsible for detecting certain types of entities in the input text document, which are about contracts. 
Once the entities are recognized, they are processed according to their type.
Some of them will be replaced by <ENTITYTYPE>, encrypted, masked with a certain character, etc.
"""
# IMPORTS
# import spacy
# import random
import json
import re
from typing import List
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import AnalyzerEngine,EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import SpacyNlpEngine, NlpArtifacts, NlpEngineProvider
from presidio_anonymizer.entities.engine import OperatorConfig
from faker import Faker
import time
from cryptography.fernet import Fernet

################### GLOBALS ##################
SPACY_MODEL_PATH = "models/custom_spacy_models/400docs"
CONTRACTS_FOLDER = 'Datasets/50docs/'
################### TEMP TEST ################
class Contract:
    def __init__(self, contract_json):
        self.document_name = contract_json['document_name']
        self.document_uuid = contract_json['document_uuid']
        #self.document_content = contract_json['document_content']
        self.document_content = self.load_document_content(contract_json['document_content'])
        self.extractions = contract_json['extractions']
        self.extractions_tree = contract_json['extractions_tree']
        self.document_split = contract_json['document_split']
        #self.annotations = contract_json['annotations']
        self.annotations = [Contract_annotation(self.document_name, annotation) for annotation in contract_json['annotations']]
        self.annotations.sort(key=lambda x: x.char_start)
        self.annotated_text = self.replace_text_with_annotations()
    
    def load_document_content(self, previous_content):
        if previous_content == 'None' or previous_content == None:
            document_path = CONTRACTS_FOLDER + self.document_uuid + '/text'
            with open(document_path, 'r', encoding="utf8") as file:
                document_content = file.read()
            return document_content
        else:
            return previous_content
    
    def replace_text_with_annotations(self):
        annotated_text = ''
        last_pos = 0
        for annotation in self.annotations:
            annotated_text += self.document_content[last_pos:annotation.char_start]
            annotated_text += f'<{annotation.label}>'
            last_pos = annotation.char_end
        annotated_text += self.document_content[last_pos:]
        return annotated_text

    
# Class declaration of a single annotatio of a certain contract

class Contract_annotation:
    def __init__(self, document_name, annotation_json):
        self.document_name = document_name
        self.start = annotation_json['start']
        self.end = annotation_json['end']
        self.char_start = annotation_json['char_start']
        self.char_end = annotation_json['char_end']
        self.label = annotation_json['label']
        if 'uuid' in annotation_json.keys():
            self.uuid = annotation_json['uuid']
        self.text = annotation_json['text']
        self.origin = annotation_json['origin']

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
            getattr(other, 'text', None) == self.text)

    def __hash__(self):
        return hash(self.text + str(self.char_start) + str(self.char_end))

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
            # Declare encryption algorythm + custom Operator
            encryption_algo = Fernet(value[1].encode("utf-8"))
            operators[key] = OperatorConfig("custom", {"lambda": lambda x: str(encryption_algo.encrypt(x.encode("utf-8")))})
            #operators[key] = OperatorConfig(operator_name="encrypt", params={'key': value[1]})
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
            operators[key] = operators[value]
    
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

def complete_annotations(text, annotations):
    to_append = []
    a_start_idxs = [annotation.start for annotation in annotations]

    for annotation in annotations:
        surface_text = text[annotation.start:annotation.end]
        occurrences_search = re.finditer(pattern=re.escape(surface_text), string=text)
        candidates_idxs = [index.start() for index in occurrences_search]
        #start_indexes.remove(annotation.start) if annotation.start in start_indexes else start_indexes
        candidates_idxs = [idx for idx in candidates_idxs if idx not in a_start_idxs]
        if(len(candidates_idxs) > 0):
            #print(f'for {[surface_text, annotation.start, annotation.end]}: {[[text[idx:idx+len(surface_text)], idx, idx+len(surface_text)] for idx in candidates_idxs]}')
            for idx in candidates_idxs:
                a_start_idxs.append(idx)
                to_append.append(
                    RecognizerResult(
                        entity_type=annotation.entity_type,
                        start=idx,
                        end=idx + len(surface_text),
                        score=annotation.score
                    )
                )
    for ta in to_append:
        annotations.append(ta)
    return annotations

def anonimizar_documento(text, es_analyzer, es_anonymizer, operators, wanted_entities):
    """
    Anonymize a spanish text. Main pipeline. It will return the annonimized text as well as the annotations.
    """
    start_time = time.time()
    annotations = es_analyzer.analyze(text=text, language='es', entities=wanted_entities)
    model_annotations = len(annotations)
    complete_annotations(text, annotations)
    extended_annotations =  len(annotations) - model_annotations
    anonymized_text = es_anonymizer.anonymize(text=text, analyzer_results=annotations, operators=operators).text
    final_annotations = []
    for annotation in annotations:
        final_annotations.append({
            'type': annotation.entity_type,
            'start': annotation.start,
            'end': annotation.end,
            'surface_text': text[annotation.start:annotation.end],
            #'score': annotation.score
        })
    
    end_time = time.time()
    exec_time = end_time - start_time
    annotations_counter = {}
    for annotation in final_annotations:
        if annotation['type'] in annotations_counter:
            annotations_counter[annotation['type']] += 1
        else:
            annotations_counter[annotation['type']] = 1
    
    metrics = {
        'exec_time': round(exec_time,6),
        'exec_time_length': round(exec_time / len(text),6),
        'model_annotations': model_annotations,
        'extended_annotations': extended_annotations,
        'total_annotations': len(final_annotations),
        'annotations_counter': annotations_counter
    }
    # Build result object
    response= {
        'text': anonymized_text,
        'annotations': final_annotations,
        'metrics': metrics
    }
    return response

def main():
    # 0. Prepare all the supported entities. Including associations of previous (default) supported entites with the custom ones.
    all_entities = ['FounderName','FounderContribution','BusinessName','FounderIDNumber','FounderAddress','FounderCityName','AdminName','AdminType','OtherContribution','CompanyAddress']
    associations = {
        'ES_NIF' : 'FounderIDNumber'
    }
    all_entities.extend(list(associations.values()))
    all_entities = list(set(all_entities))

    # 1. Build custom operators (load default actions file per entity type)
    with open('src/actions.json') as json_file:
        actions = json.load(json_file)
    custom_operators = build_operators(actions_json=actions['Contratos'], associations = associations)
    
    # 2. Input data
    with open('Datasets/50docs/contract.json') as json_file: 
        contracts_json = json.load(json_file)
    contracts = []
    for document in contracts_json['documents']:
        contracts.append(Contract(document))
    # 3. Create custom spanish analyzer and anonymizer. Anonymizer is built with custom recognizer (spacy retrained model)
    es_analyzer = create_es_analyzer(all_entities)
    es_anonymizer= create_es_anonymizer()
    
    # 4. Anonymize document
    text = contracts[1].document_content
    print(text)
    response = anonimizar_documento(text, es_analyzer, es_anonymizer, operators = custom_operators, wanted_entities = all_entities)
    for val in response['annotations']:
        print(val)
    print(response['metrics'])
    """
    for contract in contracts:
        response = anonimizar_documento(contract.document_content, es_analyzer, es_anonymizer, operators = custom_operators, wanted_entities = all_entities)
        predicted_annotations = [annotation['surface_text'] for annotation in response['annotations']]
        distinct_ents = list(set(predicted_annotations))
    
        #for dent in distinct_ents:
        for entit in distinct_ents:
            text_occurrences = contract.document_content.count(entit)
            predicted_occurrences = predicted_annotations.count(entit)
            if text_occurrences != predicted_occurrences:
                print(f'Entity: {entit} appears {text_occurrences} times in the text but only {predicted_occurrences} were predicted')
                print()

    
    # 5. Display results
    print(response['text'][:1000])
    print()
    for annotation in response['annotations']:
        print(annotation)
    """

if __name__ == '__main__':
    main()

"""
import spacy
nlp = spacy.load('models/400docs')
with open('Datasets/50docs/0b3cf9dc34e64045abed88c5d48b3b48/text', 'r', encoding='utf8') as file: 
    contract = file.read()
doc = nlp(contract)
[(ent.label_, ent.text) for ent in doc.ents]
"""

"""
def create_es_analyzer_old(all_entities):
    # Create a class inheriting from SpacyNlpEngine
    class LoadedSpacyNlpEngine(SpacyNlpEngine):
        def __init__(self, loaded_spacy_model):
            self.nlp = {"es": loaded_spacy_model}
    # Load a model a-priori
    nlp = spacy.load(SPACY_MODEL_PATH)
    # Pass the loaded model to the new LoadedSpacyNlpEngine
    loaded_nlp_engine = LoadedSpacyNlpEngine(loaded_spacy_model = nlp)
    # Pass the engine to the analyzer
    analyzer = AnalyzerEngine(nlp_engine = loaded_nlp_engine, supported_languages=["es"])
    contracts_recognizer = Contracts_recognizer(supported_entities=all_entities, supported_language= 'es')
    analyzer.registry.add_recognizer(contracts_recognizer)
    return analyzer
"""
#"Et0TDZ1h1j1synHuxBNGuvemnCZorZpXCo6McQnSFhw="