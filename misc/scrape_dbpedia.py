#!/usr/bin/env python3

import argparse
import os
import sys
import json
from multiprocessing import Pool
from SPARQLWrapper import SPARQLWrapper, JSON


PERSON_QUERY = """
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX schema: <http://schema.org/>
SELECT ?person WHERE {{?person foaf:name "{}"@en}}
"""

DATA_QUERY = """
SELECT ?property ?hasValue ?isValueOf
WHERE {{
  {{ <{url}> ?property ?hasValue }}
}}
"""

LINK_TYPES = ('wikiPageWikiLink', 'wikiPageExternalLink')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'out_dir', type=str, help='Directory to write scraped results to')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-l', '--list', dest='name_file', type=str,
        help='A text file, with one name per line')
    group.add_argument(
        '-n', '--name', type=str, help='A single name')

    parser.add_argument(
        '-q', '--query', type=str,
        help='A specific query corresponding to a specific name')
    return parser.parse_args()


def load_names(fpath):
    names = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line:
                names.append(line)
    return names


def split_uri(uri):
    idx = uri.rfind('/')
    return uri[:idx], uri[idx + 1:]


def select_uri(name, uris):
    """Use heuristics to guess which URI is the correct one"""
    if len(uris) == 1:
        return uris[0]
    plausible_uris = []
    for u in uris:
        prefix, resource = split_uri(u)
        if name.lower() in resource.lower().replace('_', ' '):
            if name.lower() == resource.lower().replace('_', ' '):
                plausible_uris.append((0, u))
            elif 'television' in resource.lower():
                plausible_uris.append((1, u))
            elif 'journalist' in resource.lower():
                plausible_uris.append((1, u))
            elif 'news' in resource.lower():
                plausible_uris.append((1, u))
            elif 'politic' in resource.lower():
                plausible_uris.append((2, u))
            else:
                plausible_uris.append((len(resource), u))
    if len(plausible_uris) == 0:
        plausible_uris.extend(uris)
    plausible_uris.sort(key=lambda x: x[0])
    return plausible_uris[0][1]


def split_upper_join(s, delim=' '):
    return delim.join(t[0].upper() + t[1:] for t in s.split(delim))


def to_name_case(name):
    name = split_upper_join(name)
    name = split_upper_join(name, "'")
    name = split_upper_join(name, '-')
    name = split_upper_join(name, 'Mac')
    return name


def query_dbpedia(name):
    print('Querying: ' + name, file=sys.stderr)
    if name.islower():
        name_cased = to_name_case(name)
        print('  name cased: ' + name_cased, file=sys.stderr)
    else:
        name_cased = name
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(PERSON_QUERY.format(name_cased.replace('"', '')))
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    uris = []
    for result in results["results"]["bindings"]:
        if 'person' in result:
            uri = result['person']['value']
            uris.append(uri)

    if len(uris) > 0:
        print('  found {} URIs:'.format(len(uris)), file=sys.stderr)

        uris.sort(key=lambda x: len(x))
        for uri in uris:
            print('    ' + uri, file=sys.stderr)
        selected_uri = select_uri(name, uris)

        print('  using:', selected_uri, file=sys.stderr)
        sparql.setQuery(DATA_QUERY.format(url=selected_uri))
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        parsed_results = []
        for result in results["results"]["bindings"]:
            r_prop = result.get('property')
            if not r_prop:
                continue
            if r_prop['type'] == 'uri':
                r_prop_value = r_prop['value']
                if r_prop_value.startswith('http://dbpedia.org/ontology/'):
                    if r_prop_value.endswith(LINK_TYPES):
                        pass
                    elif r_prop_value.endswith('abstract'):
                        pass
                    else:
                        parsed_results.append(result)
                elif r_prop_value.startswith('http://purl.org/'):
                    parsed_results.append(result)
        return {
            'name': name_cased,
            'uri': selected_uri,
            'data': parsed_results,
            'other_uris': uris
        }
    print('  no suitable URIs:'.format(len(uris), name_cased), file=sys.stderr)
    return None


def process_single_name(name, out_path):
    if not os.path.exists(out_path):
        result = query_dbpedia(name)
        if result:
            with open(out_path, 'w') as f:
                json.dump(result, f)
        return result is not None
    return True


def main(out_dir, name_file=None, name=None, query=None, n=4):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    def get_out_path(name):
        return os.path.join(out_dir, '{}.json'.format(name))

    if name_file:
        assert query is None, 'Specific queries are not allowed with a file'
        names = load_names(name_file)
        with Pool(n) as p:
            has_results = p.starmap(
                process_single_name, [(n, get_out_path(n)) for n in names])
        print('Done!', file=sys.stderr)

        names_wo_results = [a for a, b in zip(names, has_results) if not b]
        if names_wo_results:
            print('The following names are missing results:')
            for name in names_wo_results:
                print(name)
    elif name:
        result = query_dbpedia(query if query else name)
        if result:
            out_path = get_out_path(name)
            with open(out_path, 'w') as f:
                json.dump(result, f)
            print('Success!', file=sys.stderr)
        else:
            print('No entry found!... try again with a different query?',
                  file=sys.stderr)
    else:
        raise Exception('Unreachable')



if __name__ == '__main__':
    main(**vars(get_args()))
