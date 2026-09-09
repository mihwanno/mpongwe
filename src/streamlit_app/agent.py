from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from typing import Any, Optional

import utils as utls


# LOAD THE various indexes, dictionary and models in a way that is cached for streamlit


# -----------------------------
# 1. STATE DEFINITION
# -----------------------------
class TranslationState(TypedDict, total=False):
    # resources
    model: Any
    llm: Any
    grammar: str

    dictionary_fr: dict
    dictionary_mp: dict

    index_fr: Any
    index_mp: Any

    mapping_fr: dict
    mapping_mp: dict

    raponda: dict

    # input/output
    user_text: str
    source_lang: str
    target_lang: str

    tokens: List[str]
    dictionary_hits: List[dict]
    candidate_translations: List[dict]

    final_translation: str
    notes: List[str]



# -----------------------------
# 2. GRAPH NODES
# -----------------------------
# Each node takes and returns (partial) state
def node_language_detection(state: TranslationState) -> TranslationState:
    """
    LLM determines the input language
    """
    language = utls.detect_language(state['user_text'])
    if language == 'fr':
        state["source_lang"] = language
        state["target_lang"] = 'mp'
    else:
        state["source_lang"] = "mp"
        state["target_lang"] = "fr"

    print(f"\n[Node: Language Detect]: {state['source_lang']}")
    return state


def node_tokenization(state: TranslationState) -> TranslationState:
    state["tokens"] = utls.tokenizer_lemmatizer(state["user_text"])
    print(f"\n[Node: Tokenization]: {state['tokens']}")
    return state


def node_semantic_lookup(state: TranslationState) -> TranslationState:
    state["dictionary_hits"] = []

    if state["source_lang"] == "fr":
        candidates = utls.collect_entries(
                                state["model"],
                                state["raponda"],
                                state["index_fr"],
                                state["dictionary_fr"],
                                state["mapping_fr"],
                                state["tokens"])
    
    else:
        candidates = utls.collect_entries(
                                state["model"],
                                state["raponda"],
                                state["index_mp"],
                                state["dictionary_mp"],
                                state["mapping_mp"],
                                state["tokens"])

    state["dictionary_hits"] = candidates
    print(f"\n[Node: Semantic Lookup]: Found {len(state['dictionary_hits'])} entries")
    for entry in state["dictionary_hits"]:
        print(f"\n{entry}")
    return state


def node_candidate_assembly(state: TranslationState) -> TranslationState:
    """
    LLM assembles candidate translations
    using ONLY dictionary_hits + vector_hits
    """
    llm = state["llm"]

            #You can use a grammar of Mpongwe language to help you with your task.
            #Mpongwe Grammar:
            #{state['grammar']}

    prompt = f"""
            You are a linguistic expert knowlegeable in French and Mpongue.
            Your objective is to translate the user input text below from {state['source_lang']} to {state['target_lang']}:

            {state['user_text']}

            You are provided with multiple dictionary entries to help you in your task.
            Dictionary evidence:
            {state['dictionary_hits']}

            Using these dictionary entries, work out the most likely translations.

            Rules:
            - Do NOT invent vocabulary
            - Prefer literal translation
            - Produce 1–3 candidate translations

            Your response should be a list of json
            {{"translation_1": "your candidate translation",
             "translation_2": "your second candidate translation"}}
            """

    response = llm.invoke(prompt)
    # TODO: parse JSON safely
    state["candidate_translations"] = [response]
    print(f"\n[Node: Assembly]: Candidates: {state['candidate_translations']}")
    return state


def node_finalize(state: TranslationState) -> TranslationState:
    """
    Select best candidate and add notes
    """
    llm = state["llm"]

    prompt = f"""
                You are a linguistic expert knowlegeable in French and Mpongue.
                Your objective is to translate the user input text below from {state['source_lang']} to {state['target_lang']}:

                {state['user_text']}

                Here are a list of candidates for translation:
                Candidates:
                {state['candidate_translations']}

                Based on your expertise, knowledge of French and Mpongwe, select the best translation, and provide justification
                Return your response with the translation clearly indicated, and make 2 line breaks before your explanations
            """

    response = llm.invoke(prompt)
    # TODO: parse JSON
    state["final_translation"] = response
    state["notes"] = []
    return state


# -----------------------------
# 3. BUILD GRAPH
# -----------------------------

graph = StateGraph(TranslationState)

graph.add_node("detect_language", node_language_detection)
graph.add_node("tokenize", node_tokenization)
graph.add_node("lookup", node_semantic_lookup)
graph.add_node("assemble", node_candidate_assembly)
graph.add_node("finalize", node_finalize)

# edges
graph.set_entry_point("detect_language")
graph.add_edge("detect_language", "tokenize")
graph.add_edge("tokenize", "lookup")
graph.add_edge("lookup", "assemble")
graph.add_edge("assemble", "finalize")
graph.add_edge("finalize", END)

translation_agent = graph.compile()