from rdflib import Graph, Literal, Namespace, RDF, URIRef

# 1️ Cargar el grafo RDF desde un archivo o una URL
g = Graph()
g.parse("data/unesco-thesaurus.ttl", format="turtle")  # Cambia a "turtle" si el archivo es .ttl

# 2️ Definir los prefijos
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
UNESCO = Namespace("http://vocabularies.unesco.org/ontology#")
ISOTHES= Namespace("http://purl.org/iso25964/skos-thes#")

LANGUAGE = 'es'

prefix_concept = "El concepto " if LANGUAGE == 'es' else "The concept "
prefix_altLabel = ", también se conoce como " if LANGUAGE == 'es' else ", also known as "
prefix_scopeNote = ", se describe como " if LANGUAGE == 'es' else " which is described as "
prefix_group= ", pertenece al grupo " if LANGUAGE == 'es' else ", belongs to group "
prefix_broader= ". Cuyo concepto más amplio es " if LANGUAGE == 'es' else ". Whose broadest concept is "
prefix_narrower= ", siendo el concepto más específico " if LANGUAGE == 'es' else ", being the narrowest concept "
prefix_related= ", está relacionado a " if LANGUAGE == 'es' else ", it is related to "

prefix_broader_plural= ". Cuyos conceptos más amplios son " if LANGUAGE == 'es' else ". Whose broader concepts are "
prefix_narrower_plural= ", siendo los conceptos más específicos " if LANGUAGE == 'es' else ", being the narrowr concepts "
prefix_related_plural= ", está relacionado a los conceptos " if LANGUAGE == 'es' else ", It is related to the concepts "

def get_labels_as_string(subject, property:URIRef, head:str):
    for b in subject:
        for o in g.objects(b, property):
            if isinstance(o, Literal) and o.language == LANGUAGE:
                head += " " + o.value + "," 
    return head[:-1]
    pref_label = ""
    for o in g.objects(domain, SKOS.prefLabel):
        if isinstance(o, Literal) and o.language == LANGUAGE:
            pref_label += o.value


dataset= []
count= 0
# 3️ Buscar todos los sujetos que sean de tipo skos:Concept
for domain in g.subjects(RDF.type, UNESCO.Domain):
    print(f"\n🔹 Dominio: {domain}")

    #?domain a unesco:Domain
    if domain:
        domainLabel = ""
        for o in g.objects(domain, SKOS.prefLabel):
            if isinstance(o, Literal) and o.language == LANGUAGE:
                domainLabel= o.value 

        print(f"  🔹 prefLabel: {domainLabel}")
    
    #?domain skos:member ?group
    groups = list(g.objects(domain, SKOS.member))
    #?group a isothes:ConceptGroup
    if groups:
        for group in groups:
            groupLabel = ''
            for o in g.objects(group, SKOS.prefLabel):
                if isinstance(o, Literal) and o.language == LANGUAGE:
                    groupLabel = o.value
                    print(f"  🔹 group: {o.value if o.value else 'N/A'}")

            # ?group skos:member ?concept .
            concepts = list(g.objects(group, SKOS.member))
            # print("concepts_member", concepts)
            for concept in concepts:
                conceptLabel= prefix_concept
                for p in g.objects(concept, SKOS.prefLabel):
                    if isinstance(p, Literal) and p.language == LANGUAGE:
                        print(f"  🔹 concept prefLabel: {p.value}")
                        conceptLabel += p.value

                conceptAltLabel = ''
                for q in g.objects(concept, SKOS.altLabel):
                    if isinstance(q, Literal) and q.language == LANGUAGE:
                        print(f"  🔹 concept altLabel: {q.value}")
                        conceptAltLabel += q.value + ", "
                if conceptAltLabel != '':
                    conceptAltLabel = prefix_altLabel + conceptAltLabel[:-2]
                
                conceptScopeNote = ''                        
                for r in g.objects(concept, SKOS.scopeNote):
                    if isinstance(r, Literal) and r.language == LANGUAGE:
                        print(f"  🔹 concept scopeNote: {r.value}")
                        conceptScopeNote = r.value
                if conceptScopeNote != '':
                    conceptScopeNote = prefix_scopeNote + conceptScopeNote
                
                # 4️ Obtener prefLabel, broader, narrower y relate
                broader = list(g.objects(concept, SKOS.broader))
                broader_label = ''
                if len(broader) > 0:
                    head = prefix_broader_plural if len(broader) > 1 else prefix_broader
                    broader_label = get_labels_as_string(broader,SKOS.prefLabel, head) 

                    print(f"  🔹 broader: {broader_label if broader_label else 'N/A'}")
                
                # Lista de conceptos más específicos
                narrower = list(g.objects(concept, SKOS.narrower))
                narrower_label = ''
                if len(narrower) > 0:
                    head = prefix_narrower_plural if len(narrower) > 1 else prefix_narrower
                    narrower_label = get_labels_as_string(narrower,SKOS.prefLabel, head) 
                    print(f"  🔹 narower: {narrower_label if narrower_label else 'N/A'}")

                # Lista de conceptos relacionados
                related = list(g.objects(concept, SKOS.related))
                related_label = ''
                if len(related) > 0:
                    head = prefix_related_plural if len(related) > 1 else prefix_related
                    related_label = get_labels_as_string(related,SKOS.prefLabel, head) 

                    print(f"  🔹 related: {related_label if related_label else 'N/A'}")

                text = conceptLabel + conceptAltLabel + conceptScopeNote + prefix_group + domainLabel + ', ' + groupLabel + broader_label + narrower_label + related_label
                dataset.append(
                    {
                        'str': text,
                        'conceptUri': concept.toPython(),
                        'conceptLabel': conceptLabel,
                        'conceptAltLabel': conceptAltLabel,
                        'conceptScopeNote': conceptScopeNote,
                        'belongsTo': [
                            {
                                'group':{
                                    'uri': group.toPython(),
                                    'label': groupLabel
                                },
                                'domain': {
                                    'uri': domain.toPython(),
                                    'label': domainLabel
                                }
                            }
                        ],
                        'broader':{
                            'uri': broader, #if row[8] else None,
                            'label': broader_label
                        },
                        'narrower': {
                            'uri': narrower,
                            'label': narrower_label
                        },
                        'related': {
                            'uri': related,
                            'label': related_label
                        }
                    }
                )
                count += 1
                print("dataset", dataset)
                break
                break
            break
        break
    break