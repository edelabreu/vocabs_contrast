from rdflib import Graph
import json
from string import Template
from pprint import pprint

pprint('Loading the thesaurus')
g = Graph()
g.parse("../data/unesco-thesaurus.ttl")
language= 'es'

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
    """
    query_str= """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX isothes: <http://purl.org/iso25964/skos-thes#>
        PREFIX unesco: <http://vocabularies.unesco.org/ontology#>
        SELECT DISTINCT ?concept ?conceptLabel ?conceptaltLabel ?conceptaltscopeNote ?group ?groupLabel ?domain ?domainLabel
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

        }
        """
    template = Template(query_str)
    return template.substitute(lang=lang)


pprint('querying the graph')
qres = g.query(unesco_query(lang=language))
count = 0
query_format = []

pprint('formatting the result for the dataset')
for row in qres:
    if language == 'es':    
        altLabel= "" if row[2] is None else " también conocido como " + row[2] + ","
        scopeNote= "" if row[3] is None else " que se describe como " + row[3] 
        text= "El concepto " + row[1] + altLabel + scopeNote +" está relacionado a " + row[5]+", "+row[7]
    elif language == 'en':
        altLabel= "" if row[2] is None else ", also known as " + row[2] + ","
        scopeNote= "" if row[3] is None else " which is described as " + row[3] 
        text= "The concept " + row[1] + altLabel + scopeNote +" It is related to " + row[5]+", "+row[7]
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
                ]
            }
        )
        count += 1

pprint(f'quantity of results processed: {count}')

with open('../data/unesco-parser-'+language+'.json', 'w', encoding='utf-8') as f:
    json.dump(query_format, f, ensure_ascii=False, indent=4)