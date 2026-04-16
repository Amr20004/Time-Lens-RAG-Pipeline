import json
from llama_index.core import Document

def loading_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        chunks = json.load(file)

    documents = []
    for chunk in chunks:
        doc = Document(
            text = chunk['artifact_overview'].strip() if len(chunk['artifact_overview']) > 1 else "",
            metadata = {
                "title":chunk['title'],
                "historical_overview": chunk['historical_overview'],
                "material": chunk['material'],
                "was_found_at": chunk['was_found_at'],
                "width": chunk['width'],
                "length": chunk['length']
            }
        )
        documents.append(doc)

    print(f"there are {len(documents)} documents")
    return documents

# file_path = 'artifactDataset.json'
# documents = loading_dataset(file_path)
# print(documents[0])
