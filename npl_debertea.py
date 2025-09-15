import json
import re
import numpy as np
from rdflib import Graph, Namespace, RDF, Literal, URIRef, OWL
import logging
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Hugging Face Transformers for embeddings
from transformers import AutoTokenizer, AutoModel
import torch

# NLTK for preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NLPMapper")

# Download nltk data if not already present
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing utilities
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load DeBERTa model
MODEL_NAME = "microsoft/deberta-v3-small"  # Change to large if GPU available
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def preprocess(text):
    """Lowercase, remove punctuation, tokenize, remove stopwords, and lemmatize."""
    text = re.sub(r'[_\-\.]', ' ', text.lower())
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def get_embedding(text):
    """Return embedding vector using DeBERTa."""
    if not text.strip():
        return np.zeros(model.config.hidden_size)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]
        embedding = last_hidden_state.mean(dim=0).cpu().numpy()
    return embedding

def parse_json(json_file):
    """Extract keys from JSON file."""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    fields = set()

    def extract(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                fields.add(k)
                if isinstance(v, (dict, list)):
                    extract(v, path + f".{k}" if path else k)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, path)

    extract(data)
    return list(fields)

def parse_owl(owl_file):
    """Extract ontology terms from OWL file and return dictionary of name → URI."""
    g = Graph()
    try:
        g.parse(owl_file, format="xml")
    except Exception:
        try:
            g.parse(owl_file, format="turtle")
        except Exception:
            g.parse(owl_file, format="json-ld")

    terms = {}
    for s, p, o in g.triples((None, RDF.type, OWL.Class)):
        terms[s.split("#")[-1]] = str(s)
    for s, p, o in g.triples((None, RDF.type, OWL.ObjectProperty)):
        terms[s.split("#")[-1]] = str(s)
    for s, p, o in g.triples((None, RDF.type, OWL.DatatypeProperty)):
        terms[s.split("#")[-1]] = str(s)

    logger.info(f"Extracted {len(terms)} terms from OWL.")
    return terms

def map_fields(json_fields, ontology_terms_dict, threshold=0.7):
    """Map JSON fields to ontology terms using cosine similarity."""
    results = []
    ontology_embeddings = {}

    # Preprocess and embed ontology terms once (caching)
    logger.info("Generating embeddings for ontology terms...")
    for term_name, uri in ontology_terms_dict.items():
        processed = preprocess(term_name)
        embedding = get_embedding(processed)
        ontology_embeddings[term_name] = (embedding, uri)

    # Map each JSON field
    logger.info("Mapping JSON fields to ontology terms...")
    for field in json_fields:
        processed_field = preprocess(field)
        field_vector = get_embedding(processed_field)

        best_match = None
        best_score = 0.0
        best_uri = None

        for term_name, (emb, uri) in ontology_embeddings.items():
            if np.any(field_vector) and np.any(emb):
                score = cosine_similarity(field_vector.reshape(1, -1), emb.reshape(1, -1))[0][0]
            else:
                score = 0.0
            if score > best_score:
                best_score = score
                best_match = term_name
                best_uri = uri

        results.append({
            "field": field,
            "match": best_match,
            "match_uri": best_uri,
            "score": best_score,
            "mapped": best_score >= threshold
        })
    return results

def generate_owl(mapped_results, output_file):
    """Create an OWL file with semantic links to original ontology classes."""
    g = Graph()
    NS = Namespace("https://orbis-security.com/pe-malware-ontology#")
    g.bind("mapped", NS)
    g.bind("pe", NS)

    for idx, res in enumerate(mapped_results):
        individual = NS[f"Sample{idx+1}"]
        g.add((individual, RDF.type, NS.MappedEntity))
        g.add((individual, NS.originalField, Literal(res["field"])))

        if res["match_uri"]:
            g.add((individual, NS.mappedTo, URIRef(res["match_uri"])))
        else:
            g.add((individual, NS.mappedTo, Literal(res["match"] or "None")))

        g.add((individual, NS.similarityScore, Literal(res["score"])))

    g.serialize(destination=output_file, format="xml")
    logger.info(f"OWL file written to {output_file}")

def main():
    json_file = "D:/study/nlp_mapping/00a35f1e23cef590bdfd8d2d30ecd7024b0028e34433384a840bf37638647af1.json"
    owl_file = "D:/study/nlp_mapping/pe_malware_ontology.owl"
    output_file = "D:/study/nlp_mapping/mapped_output.owl"
    threshold = 0.7

    logger.info("Parsing JSON file...")
    json_fields = parse_json(json_file)

    logger.info("Parsing OWL file...")
    ontology_terms_dict = parse_owl(owl_file)

    mapped_results = map_fields(json_fields, ontology_terms_dict, threshold)

    for res in mapped_results:
        status = "✔" if res["mapped"] else "✘"
        logger.info(f"{res['field']} → {res['match']} ({res['score']:.2f}) {status}")

    logger.info("Generating output OWL file...")
    generate_owl(mapped_results, output_file)

if __name__ == "__main__":
    main()
