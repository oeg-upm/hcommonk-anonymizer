# -*- coding: utf-8 -*-

"""
Created on Wed Mar 31 10:23:50 2021

@author: Pablo
"""


'''


!pip install flask

!pip install flask-restplus


!pip install Werkzeug==0.16.1


'''

import flask_restplus
flask_restplus.__version__

import json
#from Service import NERModel
from flask import Flask,request
from flask_restplus import Api,Resource,fields
from flask_swagger_ui import get_swaggerui_blueprint
app = Flask(__name__)
api = Api(app=app,version='1.0', title='Hcommonk-anonymizer', description='Project to anonymize')


name_space = api.namespace('anonymizer', description='API to anonymize ')
                   



SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    '/swagger',
    '/swagger.json',
    config={
        'app_name': "Valkyr-ie-bio"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT,url_prefix='/swagger')

Texto = api.model('Texto', {
    'text': fields.String(required=True, description='Text to be processed', default=''),
})




def create_es_anonymizer():
    
    from presidio_anonymizer import AnonymizerEngine
    anonymizer = AnonymizerEngine()
    return anonymizer
    





def create_es_analyzer():
    # Create configuration containing engine name and models
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "es", "model_name": "es_core_news_sm"}],
    }
    
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    # Create NLP engine based on configuration
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    # the languages are needed to load country-specific recognizers 
    # for finding phones, passport numbers, etc.
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine,supported_languages=["es"])
    return analyzer

es_analyzer = create_es_analyzer()
es_anonymizer= create_es_anonymizer()

def generateBIOResponse(data,labels):

    #{'label':'O','word':'Hola'},{'label':'O','word':'Mundo'}
    List=[]

    for d,l in zip(data,labels):
        e ={ 'word':d,'label':l }
        List.append(e)
    return List





@name_space.route("/es/")
class NCIDIST(Resource):

    @api.expect(Texto)
    def post(self):
        """
        Anonymize a spanish text
        """
        data = request.json
        text = data.get('text')
        
        results = es_analyzer.analyze(text=text, language='es')
        from presidio_anonymizer.entities.engine import OperatorConfig
        operators={
           
           "DEFAULT": OperatorConfig(operator_name="mask", params={ 
                                              'masking_char': '*', 'chars_to_mask': 100, 'from_end': True})}

        anonymized_text = es_anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text


        #TheSentence= ('Clustering of missense mutations in the ataxia-telangiectasia gene in a sporadic T-cell leukaemia')

        
        

        response= {
            'text': anonymized_text
            }

        return response



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=8088)