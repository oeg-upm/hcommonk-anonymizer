import spacy
import json
import re
import random
from pathlib import Path
from spacy.training.example import Example
from spacy.util import minibatch, compounding

# Globals and hyperparameters
#CONTRACTS_FOLDER = 'Datasets/50docs/'
CONTRACTS_FOLDER = 'Datasets/400docs/'
CONTRACTS_JSON = CONTRACTS_FOLDER+'contract.json'
MODEL_PARAMS = {
    'n_iter': 100,
    'output_dir': 'models/400docs/',
    'new_model_name': 'nlp400docs',
    'dropout': 0.35,
    #'':,
    #spacy.util.minibatch(train_data, size=spacy.util.compounding(4., 32., 1.001))
}

print(f'Contracts_folder: {CONTRACTS_FOLDER}, MODEL_PARAMS: {MODEL_PARAMS}')
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

def build_annotations_dsict(contracts):
    annotations_dict = {}

    for contract in contracts:
        for annotation in contract.annotations:
            if annotation.label in annotations_dict:
                annotations_dict[annotation.label].append(annotation.text)
            else:
                annotations_dict[annotation.label] = [annotation.text]
    return annotations_dict

def prepare_train_data(contracts):
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

def train_nermodel(train_data, labels, model='es_core_news_sm', params=MODEL_PARAMS):
    n_iter = params['n_iter']
    # Setting up the pipeline and entity recognizer.
    if model is not None:
        nlp = spacy.load(model)  # load existing spacy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe('ner')

    # Add new entity labels to entity recognizer
    for i in labels:
        ner.add_label(i)
    # Inititalizing optimizer
    if model is None:
        optimizer = nlp.begin_training()
    else:
        #optimizer = nlp.entity.create_optimizer()
        optimizer = nlp.get_pipe("ner").create_optimizer()

    # Get names of other pipes to disable them during training to train # only NER and update the weights
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    print('begin training')
    # ADD TQM PROGRESS BAR OR SMTHNG
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch) 
                # new
                for text, annotations in batch:
                    # create Example
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    # Update the model
                    nlp.update([example], losses=losses, sgd=optimizer, drop=params['dropout'])
                # Updating the weights
                # nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print(f'Losses at it {itn}: {losses}')
    
    output_dir = params['output_dir']
    new_model_name = params['new_model_name']
    # Save model 
    if output_dir is not None:
        output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.meta['name'] = new_model_name  # rename model
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)


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

    # 3. Prepare the training data in spacy format
    train_data = prepare_train_data(contracts)
    
    # 4.
    model_nlp = train_nermodel(train_data, labels=all_labels, model='es_core_news_sm')

    # 5.
    
if __name__ == '__main__':
    main()