import json
import time
from rdflib import Graph
from string import Template
from pprint import pprint

pprint('Loading the thesaurus')
g = Graph()
start = time.time()
g.parse("../data/eurovoc-skos-ap-act.rdf")
end = time.time()
pprint(f"Time taken to load the thesaurus : {end - start:.4f} s")
pprint("the number of triples in the graph is "+ str(g.__len__()))
language= 'en'

def eurovoc_query(lang='es'):
    """
        This sparql query returns a list with the matches, 
        each element contains a list with 6 elements.\n
        positions | meaning\n
        0         | URI of the concept\n
        1         | concept Label\n
        2         | concept alternative Label\n
        3         | concept scopeNote \n
        4         | URI of the broader concept\n
        5         | broader Label\n
        6         | URI of the narrower concept\n
        7         | narrower Label\n
        8         | URI of the related concept\n
        9         | related Label\n
    """
    query_str= """
        prefix cdm: <http://publications.europa.eu/ontology/cdm#> 
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

        SELECT DISTINCT ?concept ?conceptLabel ?conceptaltLabel ?conceptaltscopeNote ?conceptBroader ?conceptBroaderLabel ?conceptNarrower ?conceptNarrowerLabel ?concepRelated ?concepRelatedLabel
        
        WHERE {
            ?concept a skos:Concept .
            ?concept skos:prefLabel ?conceptLabel
            FILTER(LANGMATCHES(LANG(?conceptLabel), "$lang"))

            OPTIONAL{
                # Get the altLabel property of the concept if exist, and filter by a language
                ?concept skos:altLabel ?conceptaltLabel
                FILTER(LANGMATCHES(LANG(?conceptaltLabel), "$lang"))
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

        }
        """
    template = Template(query_str)
    return template.substitute(lang=lang)


pprint('querying the graph')
start = time.time()
qres = g.query(eurovoc_query(lang=language))
end = time.time()
pprint(f"Time taken to query the thesaurus : {end - start:.4f} s")

count = 0
query_format = []

pprint('formatting the result for the dataset')
start = time.time()
for row in qres:
    if language == 'es':
        altLabel= "" if row[2] is None else ", también conocido como " + row[2]
        scopeNote= "" if row[3] is None else ", el cual se describe como  " + row[3] 
        broader= "" if row[5] is None else ", cuyo concepto más amplio es " + row[5]
        narrower= "" if row[7] is None else ", siendo " + row[7] + " el concepto más específico " 
        related= "" if row[9] is None else ", está relacionado a " + row[9]
        text = "El concepto " + row[1] + altLabel + scopeNote + broader + narrower + related,
            
    elif language == 'en':
        altLabel= "" if row[2] is None else ", also known as " + row[2]
        scopeNote= "" if row[3] is None else ", which is described as " + row[3] 
        broader= "" if row[5] is None else ", whose broadest concept is " + row[5]
        narrower= "" if row[7] is None else ", with "+ row[7] + " being the narrowest concept " 
        related= "" if row[9] is None else ", it is related to " + row[9]
        text = "The concept " + row[1] + altLabel + scopeNote + broader + narrower + related,
        
    query_format.append(
        {
            'str': text,
            'conceptUri': row[0].toPython(),
            'conceptLabel': row[1],
            'conceptAltLabel': row[2],
            'conceptScopeNote': row[3],
            'broader':{
                'uri': row[4].toPython() if row[4] else None,
                'label': row[5]
            },
            'narrower': {
                'uri': row[6].toPython() if row[6] else None,
                'label': row[7]
            },
            'related': {
                'uri': row[8].toPython() if row[8] else None,
                'label': row[9]
            }
        }
    )
    count += 1

pprint(f'quantity of results processed: {count}')

with open('../data/eurovoc-parser-'+language+'.json', mode='w', encoding='utf-8') as f:
    json.dump(query_format, f, ensure_ascii=False, indent=4)

end = time.time()
pprint(f"Time taken to process result and save it : {end - start:.4f} s")