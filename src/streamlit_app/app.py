import streamlit as st

from agent import translation_agent
import utils as utls
from langchain_ollama import OllamaLLM


FR_DICT = "data/processed/dictionaire_fr.json"
MP_DICT = "data/processed/dictionaire_mp.json"

FR_IDX = "data/processed/index_fr.faiss"
MP_IDX = "data/processed/index_mp.faiss"

from utils import OpenAILLM

def load_llm():
    return OpenAILLM(
        model="gpt-5-mini",  # or gpt-4.1, gpt-4o-mini, etc.
    )


# def load_llm(model_name="qwen3:14b"):   # "qwen3:14b", #gemma3:1b
#     return OllamaLLM(
#         model=model_name,
#         temperature=0.2,
#     )


# --------------------------------------
# Load & cache heavy resources
# --------------------------------------
@st.cache_resource
def load_resources():
    model = utls.load_model()
    llm = load_llm()

    dictionary_fr = utls.load_dictionary(FR_DICT)
    dictionary_mp = utls.load_dictionary(MP_DICT)

    index_fr = utls.load_indices(FR_IDX)
    index_mp = utls.load_indices(MP_IDX)

    mapping_fr = utls.get_mapping(dictionary_fr)
    mapping_mp = utls.get_mapping(dictionary_mp)

    raponda = utls.load_raponda()
    grammar = utls.grammar_rules()

    return {
        "model": model,
        "llm": llm,
        "dictionary_fr": dictionary_fr,
        "dictionary_mp": dictionary_mp,
        "index_fr": index_fr,
        "index_mp": index_mp,
        "mapping_fr": mapping_fr,
        "mapping_mp": mapping_mp,
        "raponda": raponda,
        "grammar": grammar,
    }


# --------------------------------------
# Streamlit UI
# --------------------------------------
st.set_page_config(page_title="French ↔ Mpongwe Translator", layout="centered")

st.title("French ↔ Mpongwe Translator")
st.write("Enter a sentence in **French or Mpongwe** and get a translation.")

resources = load_resources()

user_text = st.text_area(
    "Text to translate",
    placeholder="Ex: Je mange une mangue / Osengè limbini",
    height=120,
)

translate_clicked = st.button("Translate")

# --------------------------------------
# Run the agent
# --------------------------------------
if translate_clicked and user_text.strip():
    with st.spinner("Translating..."):
        initial_state = {
            **resources,
            "user_text": user_text.strip(),
        }

        result = translation_agent.invoke(initial_state)
        dictionary_hits = result.get("dictionary_hits", [])

    with st.sidebar:
        st.header("📚 Dictionary Evidence")

        if dictionary_hits:
            for i, entry in enumerate(dictionary_hits, 1):
                st.markdown(f"**Entry {i}**")
                st.json(entry)
        else:
            st.write("No dictionary entries found.")

    st.subheader("Translation")
    st.write(result.get("final_translation", "—"))

    if result.get("notes"):
        st.subheader("Notes")
        for note in result["notes"]:
            st.markdown(f"- {note}")

elif translate_clicked:
    st.warning("Please enter some text.")