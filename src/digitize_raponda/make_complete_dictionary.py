"""
Take all the json files and produce a complete {key: value} dictionary with
{mpongwe : (grammatical type, french), french : mpongwe, example_usage_mpongwe : example_usage_french, example_usage_french : example_usage_mpongwe}

Vectorize all keys in a vector index
"""

import json
import os
from tqdm import tqdm
import pdb

def load_file(filename):
    with open(filename, "r") as file:
        return json.load(file)

def parse_entry(entry):
    # This will be the word-word and sentence-sentence translations used for vec searches
    mpongwe = entry["mpongwe"].lower()
    french = entry["francais"].lower()
    example_usages = entry["examples_usage"]
    new_entries_mp = {}
    new_entries_fr = {}
    new_entries_mp[mpongwe] = (french, mpongwe)   # 2nd item is to point back to raponda dictionary
    new_entries_fr[french] = (mpongwe, mpongwe)
    for item in example_usages:
        try:
            new_entries_mp[item[0].lower()] = (item[1].lower(), mpongwe)
            new_entries_fr[item[1].lower()] = (item[0].lower(), mpongwe)
        # new_entries[list(item.keys())[0].lower()] = list(item.values())[0].lower()
        except:
            continue
    return new_entries_mp, new_entries_fr

def convert_entry_to_lower(entry):
    entry["mpongwe"] = entry["mpongwe"].lower()
    entry["francais"] = entry["francais"].lower()
    example_usages = []
    for item in entry["examples_usage"]:
        example_usages.append({key.lower(): val.lower() for key, val in item.items()})
    return entry


ROOT = "data/intermediate/json/"

data = []
for filename in tqdm(os.listdir(ROOT)):
    if "json" in filename:
        data.extend(load_file(ROOT + filename))

# This is raponda dictionary with mpongwe entrypoint
dictionary = {}
for item in tqdm(data):
    item = convert_entry_to_lower(item)
    dictionary[item["mpongwe"].lower()] = item
    # dictionary[item["francais"].lower()] = item

with open("data/processed/mpongwe_francais.json", "w", encoding='utf-8') as f:
    json.dump(dictionary, f,  ensure_ascii=False, indent=4)


# This would be 2 dictionary of sentence-sentence or word-translation, for vec search
# 2 separate indexes, mp & fr, that will provide mapping to mpongwe entry of dictionary
index_mp = {}
index_fr = {}
for item in tqdm(data):
    new_entries_mp, new_entries_fr = parse_entry(item)
    index_mp.update(new_entries_mp)
    index_fr.update(new_entries_fr)

with open("data/processed/dictionaire_mp.json", "w", encoding='utf-8') as f:
    json.dump(index_mp, f,  ensure_ascii=False, indent=4)

with open("data/processed/dictionaire_fr.json", "w", encoding='utf-8') as f:
    json.dump(index_fr, f,  ensure_ascii=False, indent=4)
    