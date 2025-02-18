"""

########
# This work is based on storing in a JSON 
# the result of processing, through Python,
# the SPARQL query described below.
########


PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX isothes: <http://purl.org/iso25964/skos-thes#>
PREFIX unesco: <http://vocabularies.unesco.org/ontology#>
SELECT DISTINCT ?concept ?conceptLabel ?conceptaltLabel ?conceptaltscopeNote ?group ?groupLabel ?domain ?domainLabel ?conceptBroader ?conceptBroaderLabel ?conceptNarrower ?conceptNarrowerLabel ?concepRelated ?concepRelatedLabel
WHERE {
    # Get the domains of the ontology
    ?domain a unesco:Domain .
    # From these domains obtain their members, which in this case are the intermediate groups
    ?domain skos:member ?group .
    # These members are classified as ConceptGroup
    ?group a isothes:ConceptGroup  .
    # From these groups obtain their members who are the concepts
    ?group skos:member ?concept .
    # Check that they are concepts
    ?concept a skos:Concept .

    # Get the Label property of the domains and filter by a language
    ?domain skos:prefLabel ?domainLabel
    FILTER(LANGMATCHES(LANG(?domainLabel), "$lang"))
    # Get the Label property of the groups and filter by a language
    ?group skos:prefLabel ?groupLabel
    FILTER(LANGMATCHES(LANG(?groupLabel), "$lang"))
    # Get the Label property of the concepts and filter by a language
    ?concept skos:prefLabel ?conceptLabel
    FILTER(LANGMATCHES(LANG(?conceptLabel), "$lang"))

    OPTIONAL{
        # Get the altLabel property of the concepts if exist, and filter by a language
        ?concept skos:altLabel ?conceptaltLabel
        FILTER(LANGMATCHES(LANG(?conceptaltLabel), "$lang"))
    }

    OPTIONAL{
        # Get the scopeNote property of the concepts if exist, and filter by a language
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

thesaurus_name="../data/unesco-thesaurus.ttl"
file_name= '../data/unesco-dataset-'+LANGUAGE+'.json'

# 1️ Load the RDF graph from a file or a URL.
pprint('Loading the graph')
start = time.time()
g = Graph()
g.parse(thesaurus_name, format="turtle")
end = time.time()
pprint(f"Time taken to load the thesaurus : {end - start:.4f} s")

# 2️ Define the graph prefixes.
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
UNESCO = Namespace("http://vocabularies.unesco.org/ontology#")
ISOTHES= Namespace("http://purl.org/iso25964/skos-thes#")

# Setting headers according to language
prefix_concept = "El concepto " if LANGUAGE == 'es' else "The concept "
prefix_altLabel = ", también se conoce como " if LANGUAGE == 'es' else ", also known as "
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
#?domain a unesco:Domain
for domain in g.subjects(RDF.type, UNESCO.Domain):

    if domain:
        domainLabel, domainLabel_array = get_labels(domain)
    
    #?domain skos:member ?group
    groups = list(g.objects(domain, SKOS.member))
    
    if groups:
        for group in groups:
            groupLabel, groupLabel_array = get_labels(group)

            # ?group skos:member ?concept .
            concepts = list(g.objects(group, SKOS.member))
            
            for concept in concepts:
                text = ''
                # get the concept prefLabel
                conceptLabel, conceptLabel_array = get_labels(concept)
                text += prefix_concept + conceptLabel
                
                # get the concept altfLabel
                conceptAltLabel, conceptAltLabel_array = get_labels(concept, SKOS.altLabel)
                if conceptAltLabel != '':
                    text += prefix_altLabel + conceptAltLabel
                
                # add domain and group to the final text
                text += prefix_group + domainLabel + ', ' + groupLabel
                
                # get the concept ScopeNote
                conceptScopeNote, conceptScopeNote_array = get_labels(concept, SKOS.scopeNote)
                if conceptScopeNote != '':
                    text += prefix_scopeNote + conceptScopeNote
                
                ### obtains the concepts broader, narrower and related
                
                broader = list(g.objects(concept, SKOS.broader))
                broader_label = ''
                if len(broader) > 0:
                    broader_label, broader_label_array, broader_uri_array = get_labels_from_array(broader,SKOS.prefLabel, '') 
                    text += prefix_broader_plural + broader_label if len(broader) > 1 else prefix_broader + broader_label
                
                narrower = list(g.objects(concept, SKOS.narrower))
                narrower_label = ''
                if len(narrower) > 0:
                    narrower_label, narrower_label_array, narrower_uri_array = get_labels_from_array(narrower,SKOS.prefLabel, '')
                    text += prefix_narrower_plural + narrower_label if len(narrower) > 1 else prefix_narrower + narrower_label

                related = list(g.objects(concept, SKOS.related))
                related_label = ''
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
                        'belongsTo': [
                            {
                                'group':{
                                    'uri': group.toPython(),
                                    'label': groupLabel_array
                                },
                                'domain': {
                                    'uri': domain.toPython(),
                                    'label': domainLabel_array
                                }
                            }
                        ],
                        'broader':{
                            'uri': broader_uri_array, #if row[8] else None,
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
