(Under Development)
This project implements a robust Natural Language Processing (NLP) pipeline to map fields from structured JSON data to classes and properties defined in an OWL ontology. By leveraging advanced techniques such as tokenization, lemmatization, stopword removal, and semantic embeddings from spaCy, the system identifies the best-matching ontology terms based on contextual similarity rather than exact string matches.

The output is a semantically enriched OWL file where each JSON field is linked to its corresponding ontology class or property using RDF relationships. These links include similarity scores to indicate the confidence of each mapping. The resulting file ensures interoperability and traceability, making it suitable for reasoning frameworks like ECII and other semantic web applications.

Key features include:
 Automated extraction of JSON fields and ontology terms
 Semantic similarity comparison using word embeddings
 Confidence scoring for mappings
 Semantic links referencing original ontology terms
 Extensible, reusable pipeline for various domains

This approach significantly enhances data integration and interpretation, especially in scenarios where structured data must align with complex domain ontologies.
