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
MODEL_NAME = "microsoft/deberta-v3-small"  # Use 'large' if GPU available
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

def extract_value(data, field_name):
    """Recursively search for the field value in the JSON data."""
    if isinstance(data, dict):
        for k, v in data.items():
            if k == field_name:
                return v
            elif isinstance(v, (dict, list)):
                result = extract_value(v, field_name)
                if result is not None:
                    return result
    elif isinstance(data, list):
        for item in data:
            result = extract_value(item, field_name)
            if result is not None:
                return result
    return None

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

    logger.info("Generating embeddings for ontology terms...")
    for term_name, uri in ontology_terms_dict.items():
        processed = preprocess(term_name)
        embedding = get_embedding(processed)
        ontology_embeddings[term_name] = (embedding, uri)

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

def generate_updated_owl(mapped_results, json_file, owl_file, output_file):
    """Update the OWL file by adding instances for mapped fields."""
    g = Graph()
    try:
        g.parse(owl_file, format="xml")
    except Exception:
        try:
            g.parse(owl_file, format="turtle")
        except Exception:
            g.parse(owl_file, format="json-ld")

    NS = Namespace("https://orbis-security.com/pe-malware-ontology#")
    g.bind("mapped", NS)
    g.bind("pe", NS)

    # Load JSON data
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, res in enumerate(mapped_results):
        term_uri = res["match_uri"]
        if not term_uri or not res["mapped"]:
            continue  # Skip unmapped or low-confidence fields

        term_ref = URIRef(term_uri)

        # Create a unique instance URI
        instance_uri = URIRef(f"{term_uri}_Instance{idx+1}")

        # Declare it as an instance of the ontology term
        g.add((instance_uri, RDF.type, term_ref))

        # Add additional properties
        g.add((instance_uri, NS.fieldName, Literal(res["field"])))
        g.add((instance_uri, NS.similarityScore, Literal(res["score"])))

        value = extract_value(data, res["field"])
        if value is not None:
            g.add((instance_uri, NS.fieldValue, Literal(str(value))))

    g.serialize(destination=output_file, format="xml")
    logger.info(f"Updated OWL with instances saved to {output_file}")

def main():
    json_file = "D:/study/nlp_mapping/00a35f1e23cef590bdfd8d2d30ecd7024b0028e34433384a840bf37638647af1.json"
    owl_file = "D:/study/nlp_mapping/pe_malware_ontology.owl"
    output_file = "D:/study/nlp_mapping/pe_malware_ontology_updated.owl"
    threshold = 0.7

    logger.info("Parsing JSON file...")
    json_fields = parse_json(json_file)

    logger.info("Parsing OWL file...")
    ontology_terms_dict = parse_owl(owl_file)

    mapped_results = map_fields(json_fields, ontology_terms_dict, threshold)

    for res in mapped_results:
        status = "✔" if res["mapped"] else "✘"
        logger.info(f"{res['field']} → {res['match']} ({res['score']:.2f}) {status}")

    logger.info("Generating updated OWL file with instances...")
    generate_updated_owl(mapped_results, json_file, owl_file, output_file)

if __name__ == "__main__":
    main()
