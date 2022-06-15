"""
FounderIDNumber: mask 8x*
FounderName: None
FounderAddress: replace <FounderAddress>
FounderCityName: Random Replace
FounderContribution: mask 8x*
OtherContribution: Encrypt "Et0TDZ1h1j1synHuxBNGuvemnCZorZpXCo6McQnSFhw="
CompanyAddress: replace <CompanyAddress>
BusinessName: None
"""

from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer.entities.engine import OperatorConfig

def create_es_anonymizer():
    anonymizer = AnonymizerEngine()
    return anonymizer
    
def create_es_analyzer():
    # Create configuration containing engine name and models
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "es", "model_name": "es_core_news_sm"}],
    }    
    # Create NLP engine based on configuration
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    # the languages are needed to load country-specific recognizers 
    # for finding phones, passport numbers, etc.
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine,supported_languages=["es"])
    return analyzer

def analizar(text, es_analyzer, es_anonymizer):
    """
    Anonymize a spanish text
    """
    results = es_analyzer.analyze(text=text, language='es')
    
    operators={
        "DEFAULT": OperatorConfig(operator_name="mask", params={ 
                                            'masking_char': '*', 'chars_to_mask': 100, 'from_end': True})}
    
    anonymized_text = es_anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text

    #TheSentence= ('Clustering of missense mutations in the ataxia-telangiectasia gene in a sporadic T-cell leukaemia')
    response= {
        'text': anonymized_text
        }
    return response

def main():
    text = "Hola soy Juan, trabajo en CemenPalma y mi número de teléfono es 678 987 635. Vivo en tenerife y soy católico"
    es_analyzer = create_es_analyzer()
    es_anonymizer= create_es_anonymizer()
    response = analizar(text, es_analyzer, es_anonymizer)
    print(response)

if __name__ == '__main__':
    main()