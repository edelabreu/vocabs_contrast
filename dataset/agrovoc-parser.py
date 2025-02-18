"""

########
# This work is based on storing in a JSON 
# the result of processing, through Python,
# the SPARQL query described below.
########


PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX agrovoc: <http://aims.fao.org/aos/agrovoc/>

SELECT DISTINCT ?concept ?conceptLabel ?conceptaltLabel ?conceptaltscopeNote ?conceptBroader ?conceptBroaderLabel ?conceptNarrower ?conceptNarrowerLabel ?concepRelated ?concepRelatedLabel ?group ?groupLabel

WHERE {
    ?concept a skos:Concept .
    ?concept skos:prefLabel ?conceptLabel
    FILTER(LANGMATCHES(LANG(?conceptLabel), "$lang"))

    OPTIONAL{
        # Get the altLabel property of the concept if exist, and filter by a language
        ?concept skos:altLabel ?conceptaltLabel
        FILTER(LANGMATCHES(LANG(?conceptaltLabel), "$lang"))
    }

    OPTIONAL {
        # Get the definition property of the concept if exist, and filter by a language
        ?concept skos:definition ?definition .
        ?definition ?property ?textDefinition .
        FILTER(LANG(?textDefinition) = "$lang")  
    }

    OPTIONAL{
        # Get the scopeNote property of the concept if exist, and filter by a language
        ?concept skos:scopeNote ?conceptaltscopeNote
        FILTER(LANGMATCHES(LANG(?conceptaltscopeNote), "$lang"))
    }
    
    OPTIONAL
    {
        # Get the broader concept if exist, and filter by a language
        ?concept skos:broader ?conceptBroader .
        
        ?conceptBroader skos:prefLabel ?conceptBroaderLabel
        FILTER(LANGMATCHES(LANG(?conceptBroaderLabel), "$lang"))
    }

    OPTIONAL
    {
        # Get the narrower concept if exist, and filter by a language
        ?concept skos:narrower ?conceptNarrower .
        
            ?conceptNarrower skos:prefLabel ?conceptNarrowerLabel
        FILTER(LANGMATCHES(LANG(?conceptNarrowerLabel), "$lang"))
    }

    OPTIONAL
    {
        # Get the related concept if exist, and filter by a language
        ?concepRelated skos:related ?concept .

        ?concepRelated skos:prefLabel ?concepRelatedLabel
        FILTER(LANGMATCHES(LANG(?concepRelatedLabel), "$lang"))
    }
    OPTIONAL
    {
        ?group a skos:Collection .
        ?group skos:member ?concept .
        
        ?group skos:prefLabel ?groupLabel
        FILTER(LANGMATCHES(LANG(?groupLabel), "$lang"))
    }

}
"""

import json, time, sys
from pprint import pprint
from rdflib import Graph, Literal, Namespace, RDF, URIRef

LANGUAGE = ''
# Gets the language parameter
if len(sys.argv) > 1:
    LANGUAGE = sys.argv[1]
    print(f"Received parameter: {LANGUAGE}")
    if LANGUAGE not in ['es', 'en']:
        raise("Parameter is not correct. You must provide one of the following possible parameters ['es', 'en']")
else:
    raise("No parameter was provided. You must provide one of the following possible parameters ['es', 'en']")

thesaurus_name="../data/agrovoc_lod.rdf"
file_name= '../data/agrovoc-dataset-'+LANGUAGE+'.json'

# 1️ Load the RDF graph from a file or a URL.
pprint('Loading the graph')
start = time.time()
g = Graph()
g.parse(thesaurus_name, format="xml")
end = time.time()
pprint(f"Time taken to load the thesaurus : {end - start:.4f} s")

# 2️ Define the graph prefixes.
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
AGROVOC = Namespace("http://aims.fao.org/aos/agrovoc/")

# Setting headers according to language
prefix_concept = "El concepto " if LANGUAGE == 'es' else "The concept "
prefix_altLabel = ", también se conoce como " if LANGUAGE == 'es' else ", also known as "
prefix_definition = ". Su definición es " if LANGUAGE == 'es' else ". Its definition is "
prefix_scopeNote = ". Se describe como " if LANGUAGE == 'es' else ". Which is described as "
prefix_group= ". Pertenece al grupo " if LANGUAGE == 'es' else ". It belongs to group "
prefix_broader= ". Cuyo concepto más amplio es " if LANGUAGE == 'es' else ". Whose broadest concept is "
prefix_narrower= ". Siendo el concepto más específico " if LANGUAGE == 'es' else ". Being the narrowest concept "
prefix_related= ". Está relacionado a " if LANGUAGE == 'es' else ". It is related to "

prefix_broader_plural= ". Cuyos conceptos más amplios son " if LANGUAGE == 'es' else ". Whose broader concepts are "
prefix_narrower_plural= ". Siendo los conceptos más específicos " if LANGUAGE == 'es' else ". Being the narrowr concepts "
prefix_related_plural= ". Está relacionado a los conceptos " if LANGUAGE == 'es' else ". It is related to the concepts "


def get_labels_from_array(subject, predicate:URIRef, head:str):
    """
        Iterates through a list of data and 
        evaluates each element based on the specified predicate.
    """
    array = []
    array_uri= []
    for b in subject:
        array_uri.append(b.toPython())
        for o in g.objects(b, predicate):
            if isinstance(o, Literal) and o.language == LANGUAGE:
                head += o.value + ", " 
                array.append(o.value)
    return head[:-2], array, array_uri

def get_labels(element, predicate:URIRef= SKOS.prefLabel):
    """Evaluates each element based on the specified predicate."""
    labels = ''
    labels_array = []
    for o in g.objects(element, predicate):
        if isinstance(o, Literal) and o.language == LANGUAGE:
            labels += o.value + ', '
            labels_array.append(o.value)
    return labels[:-2], labels_array

dataset= []
count= 0

pprint('querying the graph')
start = time.time()


# ?group skos:member ?concept .
concepts = list(g.subjects(RDF.type, SKOS.Concept))

for concept in concepts:
    text = ''
    # get the concept prefLabel
    conceptLabel, conceptLabel_array = get_labels(concept)
    text += prefix_concept + conceptLabel
    
    # get the concept altfLabel
    conceptAltLabel, conceptAltLabel_array = get_labels(concept, SKOS.altLabel)
    if conceptAltLabel != '':
        text += prefix_altLabel + conceptAltLabel
    
    # Recorrer los conceptos y sus definiciones
    # ?concept skos:definition ?definition .
    # ?definition ?property ?textDefinition .
    # FILTER (LANG(?textDefinition) = "es")
    definition = list(g.objects(concept, SKOS.definition))
    definition_label = ''
    definition_label_array=[] 
    definition_uri_array=[]
    if len(definition) > 0:
        for d in definition:
            definition_uri_array.append(d.toPython())
            for p, o in g.predicate_objects(subject=d):
                if isinstance(o, Literal) and o.language == LANGUAGE:
                    definition_label += o + ', '
                    definition_label_array.append(o) 
                    text += prefix_definition + definition_label[:-2]
    
    groups = list(g.subjects(SKOS.member, concept))
    groups_label = []
    groups_label_array =[]
    groups_uri_array = []
    if len(groups) > 0:
        groups_label, groups_label_array, groups_uri_array = get_labels_from_array(groups,SKOS.prefLabel, '') 
        text += prefix_group + groups_label if len(groups) > 1 else prefix_group + groups_label
    
    # get the concept ScopeNote
    conceptScopeNote, conceptScopeNote_array = get_labels(concept, SKOS.scopeNote)
    if conceptScopeNote != '':
        text += prefix_scopeNote + conceptScopeNote
    
    ### obtains the concepts broader, narrower and related
    
    broader = list(g.objects(concept, SKOS.broader))
    broader_label = ''
    broader_label_array=[] 
    broader_uri_array=[]
    if len(broader) > 0:
        broader_label, broader_label_array, broader_uri_array = get_labels_from_array(broader,SKOS.prefLabel, '') 
        text += prefix_broader_plural + broader_label if len(broader) > 1 else prefix_broader + broader_label
    
    narrower = list(g.objects(concept, SKOS.narrower))
    narrower_label = ''
    narrower_label_array=[] 
    narrower_uri_array=[]
    if len(narrower) > 0:
        narrower_label, narrower_label_array, narrower_uri_array = get_labels_from_array(narrower,SKOS.prefLabel, '')
        text += prefix_narrower_plural + narrower_label if len(narrower) > 1 else prefix_narrower + narrower_label

    related = list(g.objects(concept, SKOS.related))
    related_label = ''
    related_label_array=[] 
    related_uri_array=[]
    if len(related) > 0:
        related_label, related_label_array, related_uri_array = get_labels_from_array(related,SKOS.prefLabel, '') 
        text += prefix_related_plural + related_label if len(related) > 1 else prefix_related + related_label

    dataset.append(
        {
            'str': text,
            'conceptUri': concept.toPython(),
            'conceptLabel': conceptLabel_array,
            'conceptAltLabel': conceptAltLabel_array,
            'conceptScopeNote': conceptScopeNote_array,
            'conceptDefinition': {
                'uri': definition_uri_array,
                'label': definition_label_array
            },
            'belongsTo': [
                {
                    'group':{
                        'uri': groups_uri_array,
                        'label': groups_label_array
                    }
                }
            ],
            'broader':{
                'uri': broader_uri_array,
                'label': broader_label_array
            },
            'narrower': {
                'uri': narrower_uri_array,
                'label': narrower_label_array
            },
            'related': {
                'uri': related_uri_array,
                'label': related_label_array
            }
        }
    )
    count += 1
    # print("dataset", dataset)


end = time.time()
pprint(f"Time taken to query the thesaurus and format the result: {end - start:.4f} s")

pprint(f'Quantity of results processed: {count}')

pprint("Saving dataset")
with open(file_name, mode='w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)
