"""
Script to format the data for a spacy model
"""

# https://towardsdatascience.com/custom-named-entity-recognition-using-spacy-7140ebbb3718
# https://towardsdatascience.com/train-ner-with-custom-training-data-using-spacy-525ce748fab7
# https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/
import spacy
import json
import re
import random
from pathlib import Path
from spacy.training.example import Example
from spacy.util import minibatch, compounding
from spacy.tokens import DocBin
import warnings

# Globals and hyperparameters
# 50docs or 400docs
CONTRACTS_FOLDER = 'Datasets/400docs/'
CONTRACTS_JSON = CONTRACTS_FOLDER+'contract.json'
OUTPUT_PATH = 'Datasets/spacy_data/400docs/'

print(f'Contracts_folder: {CONTRACTS_FOLDER}, output_folder: {OUTPUT_PATH}')
input1 = input('Check if all correct, then press 1: ')
if input1 != '1':
    print('Not pressed key 1, closing.')
    exit()

# Class declaration of a single contract
class Contract:
    def __init__(self, contract_json):
        self.document_name = contract_json['document_name']
        self.document_uuid = contract_json['document_uuid']
        #self.document_content = contract_json['document_content']
        self.document_content = self.load_document_content(contract_json['document_content'])
        #self.document_content = self.load_document_contentPDF(contract_json['document_content'])
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
    
    def load_document_contentPDF(self, previous_content):
        if previous_content == 'None':
            document_path = CONTRACTS_FOLDER + self.document_uuid + '/Original.pdf'
            file = open(document_path, 'rb')
            fileReader = PyPDF2.PdfFileReader(file)
            document_content = []
            for page in range(0, fileReader.numPages):
                document_content.append(fileReader.pages[page].extractText())
            document_content = ' '.join(document_content)
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

    def clean_document_content(self):
        "!!!: the annotations will not match the document if the preprocessing/cleaning of the text is performed"
        pass
        # remove random generated lines (---------, --pooompopooo-, etc.)
        #self.document_content = re.sub('\. -+[a-zA-Z]*-+', '. ', self.document_content)

        # remove random generated lines with =
        #self.document_content = re.sub('\. [-|=|—| |\>|\+]+','. ', self.document_content)

        # remove cutted words (dieci-\nsiete)
        #self.document_content = re.sub('([a-z])-\n([a-zA-Z])', '\g<1>'+'\g<2>', self.document_content)

        #self.document_content = re.sub('(?<!\s)\\n','', self.document_content)

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

def complete_annotations(contracts):
    for contract in contracts:
        to_append = []
        a_start_idxs = [annotation.start for annotation in contract.annotations]
        text = contract.document_content
        
        for annotation in contract.annotations:
            surface_text = annotation.text
            occurrences_search = re.finditer(pattern=surface_text, string=text)
            candidates_idxs = [index.start() for index in occurrences_search]
            #start_indexes.remove(annotation.start) if annotation.start in start_indexes else start_indexes
            candidates_idxs = [idx for idx in candidates_idxs if idx not in a_start_idxs]
            if(len(candidates_idxs) > 0):
                #print(f'for {[surface_text, annotation.start, annotation.end]}: {[[text[idx:idx+len(surface_text)], idx, idx+len(surface_text)] for idx in candidates_idxs]}')
                for idx in candidates_idxs:
                    a_start_idxs.append(idx)
                    to_append.append(
                        Contract_annotation(contract.document_name, {
                            'start': -1,
                            'end': -1,
                            'char_start': idx,
                            'char_end': idx + len(surface_text),
                            'label': annotation.label,
                            'text': surface_text,
                            'origin': 'null',
                        })
                    )
                    
        for ta in to_append:
            contract.annotations.append(ta)
        contract.annotations.sort(key=lambda x: x.char_start)
    return contracts

def prepare_train_data_spacy(contracts):
    """
    TRAIN_DATA = [
        ('Who is Nishanth?', {'entities': [(7, 15, 'PERSON')]}),
        ('I like London and Berlin.', {'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]})
    ]
    """
    train_data = []
    for contract in contracts:
        entities = [(annotation.char_start, annotation.char_end, annotation.label) for annotation in contract.annotations]
        train_data.append((contract.document_content, {'entities' : entities}))
    return train_data

def convert(lang: str, training_data, output_path: Path):
    nlp = spacy.blank(lang)
    db = DocBin()
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annotations["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is None:
                msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
                warnings.warn(msg)
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    db.to_disk(output_path)

def main():
    # 1. Load contrats json file
    with open(CONTRACTS_JSON) as json_file:
        contracts_json = json.load(json_file)
    
    # 2. Load all the contracts from CONTRACTS_FOLDER 
    contracts = []
    for document in contracts_json['documents']:
        contracts.append(Contract(document))
    #   2.1 clean data
    #[contract.clean_document_content() for contract in contracts]
    #   2.1 add unlabeled annotations
    contracts = complete_annotations(contracts)
    #   2.2 remove collisions
    print(f'Total number of annotations before removing collisions: {sum([len(contract.annotations) for contract in contracts])}')
    for contract in contracts:
        final_annotations = [contract.annotations[0]]
        for i in range(1,len(contract.annotations)):
            previous_annotation = contract.annotations[i-1]
            current_annotation = contract.annotations[i]
            if current_annotation.char_start >= previous_annotation.char_end:
                # no collision
                final_annotations.append(current_annotation)
            else:
                # there is a collision, what should we do?
                # currently we are removing it
                pass
        contract.annotations = final_annotations
    print(f'Total number of annotations after removing collisions: {sum([len(contract.annotations) for contract in contracts])}')

        # remove exact collisions, what about partial collisions?
        # contract.annotations = list(set(contract.annotations))
    #   2.3 extract all labels
    all_labels = list(set([annotation.label for contract in contracts for annotation in contract.annotations]))

    # 3. Split data and prepare the training data in spacy format
    train_contracts = [contract for contract in contracts if contract.document_split == 'TRAINING']
    test_contracts = [contract for contract in contracts if contract.document_split == 'TEST']
    none_contracts = [contract for contract in contracts if contract.document_split == 'None']
    validation_contracts = [contract for contract in contracts if contract.document_split == 'VALIDATION']
    print(len(train_contracts), len(test_contracts), len(none_contracts), len(validation_contracts))
    train_contracts.extend(none_contracts)
    train_data = prepare_train_data_spacy(train_contracts)
    test_data = prepare_train_data_spacy(test_contracts)
    validation_data = prepare_train_data_spacy(validation_contracts)

    # 4.
    convert(lang = 'es', training_data = train_data, output_path = OUTPUT_PATH+'train.spacy')
    convert(lang = 'es', training_data = test_data, output_path = OUTPUT_PATH+'test.spacy')
    convert(lang = 'es', training_data = validation_data, output_path = OUTPUT_PATH+'validation.spacy')

    # 5.
    
if __name__ == '__main__':
    main()

# python -m spacy init fill-config .\Datasets\spacy_data\base_config.cfg .\Datasets\spacy_data\config.cfg
# python -m spacy train .\Datasets\spacy_data\config.cfg --output ./models/spacy_models_cl/50docs/ --paths.train .\Datasets\spacy_data\50docs\train.spacy --paths.dev .\Datasets\spacy_data\50docs\validation.spacy