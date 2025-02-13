import json
import time
from rdflib import Graph
from string import Template
from pprint import pprint

language= 'en'
thesaurus_name="../data/unesco-thesaurus.ttl"
file_name= '../data/unesco-parser-'+language+'.json'

pprint('Loading the thesaurus')
start = time.time()
g = Graph()
g.parse(thesaurus_name)
end = time.time()
pprint(f"Time taken to load the thesaurus : {end - start:.4f} s")
pprint("the number of triples in the graph is "+ str(g.__len__()))

def unesco_query(lang='es'):
    """
        This sparql query returns a list with the matches, 
        each element contains a list with 6 elements.\n
        positions | meaning\n
        0         | URI of the concept\n
        1         | concept Label\n
        2         | concept alternative Label\n
        3         | concept scopeNote \n
        4         | URI of the group that qualifies the concept\n
        5         | group Label\n
        6         | URI of the domain that qualifies the group\n
        7         | domain Label\n
        8         | URI of concept Broader
        9         | concept BroaderLabel
        10        | URI of concept Narrower 
        11        | concept Narrower Label 
        12        | URI of concep Related 
        13        | concep Related Label
    """
    query_str= """
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
    template = Template(query_str)
    return template.substitute(lang=lang)


pprint('querying the graph')
start = time.time()
qres = g.query(unesco_query(lang=language))
end = time.time()
pprint(f"Time taken to query the thesaurus : {end - start:.4f} s")

count = 0
query_format = []

pprint('formatting the result for the dataset')
start = time.time()
for row in qres:
    if language == 'es':    
        altLabel= "" if row[2] is None else ", también conocido como " + row[2] + ","
        scopeNote= "" if row[3] is None else ", que se describe como " + row[3] 
        group= "" if row[5] is None and row[7] else ", pertenece al grupo " + row[5] + " y " + row[7]
        broader= "" if row[9] is None else ", cuyo concepto más amplio es " + row[9]
        narrower= "" if row[11] is None else ", siendo " + row[11] + " el concepto más específico " 
        related= "" if row[13] is None else ", está relacionado a " + row[13]
        text= "El concepto " + row[1] + altLabel + scopeNote + group + broader + narrower + related
    elif language == 'en':
        altLabel= "" if row[2] is None else ", also known as " + row[2] + ","
        scopeNote= "" if row[3] is None else " which is described as " + row[3]
        group= "" if row[5] is None and row[7] else ", belongs to group " + row[5] + " and " + row[7]
        broader= "" if row[9] is None else ", whose broadest concept is " + row[9]
        narrower= "" if row[11] is None else ", with "+ row[11] + " being the narrowest concept " 
        related= "" if row[13] is None else ", it is related to " + row[13]
        text= "The concept " + row[1] + altLabel + scopeNote + group + broader + narrower + related
    query_format.append(
            {
                'str': text,
                'conceptUri': row[0].toPython(),
                'conceptLabel': row[1],
                'conceptAltLabel': row[2],
                'conceptScopeNote': row[3],
                'belongsTo': [
                    {
                        'group':{
                            'uri': row[4].toPython(),
                            'label': row[5]
                        },
                        'domain': {
                            'uri': row[6].toPython(),
                            'label': row[7]
                        }
                    }
                ],
                'broader':{
                    'uri': row[8].toPython() if row[8] else None,
                    'label': row[9]
                },
                'narrower': {
                    'uri': row[10].toPython() if row[10] else None,
                    'label': row[11]
                },
                'related': {
                    'uri': row[12].toPython() if row[12] else None,
                    'label': row[13]
                }
            }
        )
    count += 1

pprint(f'quantity of results processed: {count}')

with open(file_name, mode='w', encoding='utf-8') as f:
    json.dump(query_format, f, ensure_ascii=False, indent=4)

end = time.time()
pprint(f"Time taken to process result and save it : {end - start:.4f} s")