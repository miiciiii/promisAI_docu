import pandas as pd
import json
from itertools import islice
import re
from nltk.corpus import wordnet
from keybert import KeyBERT
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


kw_model = KeyBERT()

# Function to get synonyms from WordNet for a phrase (single words)
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            if synonym != word.lower():
                synonyms.add(synonym)
    return synonyms

# Function to expand keyword list with synonyms
def expand_keywords(keyword_dict):
    expanded = {}
    for label, keywords in keyword_dict.items():
        expanded_keywords = set()
        for kw in keywords:
            expanded_keywords.add(kw.lower())
            syns = get_synonyms(kw)
            expanded_keywords.update(syns)
        expanded[label] = list(expanded_keywords)
    return expanded

# Function to extract keywords using KeyBERT (instantiate model inside function)
def extract_keywords(text, top_n=10):
    keywords = kw_model.extract_keywords(text, top_n=top_n)
    return [kw for kw, _ in keywords]

# Function to assign labels based on keyword matches and extracted keywords
def assign_labels(text, keyword_dict, top_n=10):
    labels = []
    text_lower = text.lower()
    
    # Direct match from expanded keywords to text
    for label, keywords in keyword_dict.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                labels.append(label)
                break
    
    # Extract keywords from abstract
    extracted_keywords = extract_keywords(text, top_n)
    
    # Expand extracted keywords with synonyms
    expanded_extracted_keywords = set()
    for kw in extracted_keywords:
        expanded_extracted_keywords.add(kw.lower())
        expanded_extracted_keywords.update(get_synonyms(kw))
    
    # Match expanded extracted keywords to expanded keywords dict
    for ekw in expanded_extracted_keywords:
        for label, keywords in keyword_dict.items():
            if ekw in keywords:
                labels.append(label)
                break
    
    return labels


# Apply labeling in parallel (must be called inside __main__)
def parallel_apply(func, data, *args):
    results = []
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(func, data, *args), total=len(data)):
            results.append(result)
    return results

if __name__ == '__main__':

    import nltk
    
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    # Load dataset
    with open('dataset/raw/arxiv-metadata-oai-snapshot.json', 'r') as f:
        df = pd.DataFrame(json.loads(line) for line in islice(f, 10))
    print(f"Loaded {len(df)} records.")

    df = df[['id', 'title', 'abstract']].copy()
    df.dropna(subset=['abstract'], inplace=True)

    df['concept'] = df['abstract'].apply(
        lambda x: [kw for kw, score in kw_model.extract_keywords(x, top_n=10)]
    )

    # Define original keyword sets
    sdg_keywords = {
        "No Poverty": ["poverty", "income inequality", "low income"],
        "Zero Hunger": ["hunger", "malnutrition", "food security"],
        "Good Health and Well-Being": ["health", "disease", "well-being", "mental health"],
        "Quality Education": ["education", "literacy", "schooling"],
        "Gender Equality": ["gender equality", "women empowerment", "gender discrimination"],
        "Clean Water and Sanitation": ["clean water", "sanitation", "water quality"],
        "Affordable and Clean Energy": ["clean energy", "renewable energy", "sustainable energy"],
        "Decent Work and Economic Growth": ["economic growth", "employment", "job creation"],
        "Industry, Innovation and Infrastructure": ["innovation", "infrastructure", "industrialization"],
        "Reduced Inequality": ["inequality", "social disparity", "income gap"],
        "Sustainable Cities and Communities": ["sustainable cities", "urban development", "community planning"],
        "Responsible Consumption and Production": ["responsible consumption", "sustainable production", "waste reduction"],
        "Climate Action": ["climate change", "global warming", "carbon emissions"],
        "Life Below Water": ["marine life", "ocean conservation", "aquatic ecosystems"],
        "Life on Land": ["biodiversity", "deforestation", "land degradation"],
        "Peace, Justice and Strong Institutions": ["justice", "peace", "institutional development"],
        "Partnerships for the Goals": ["partnerships", "collaboration", "global cooperation"]
    }

    rd_keywords = {
        "National Integrated Basic Research Agenda": ["basic research", "fundamental studies", "theoretical research"],
        "Health": ["health", "medical", "disease", "well-being"],
        "Agriculture, Aquatic and Natural Resources": ["agriculture", "fisheries", "natural resources"],
        "Industry, Energy and Emerging Technology": ["industry", "energy", "emerging technology"],
        "Disaster Risk Reduction and Climate Change Adaptation": ["disaster risk", "climate change", "resilience"]
    }

    dost_keywords = {
        "Promotion of Human Well-Being": ["human well-being", "quality of life", "social welfare"],
        "Wealth Creation": ["wealth creation", "economic development", "income generation"],
        "Wealth Protection": ["wealth protection", "asset management", "financial security"],
        "Sustainability": ["sustainability", "environmental conservation", "sustainable development"]
    }

    agenda_keywords = {
        "Food Security": ["food security", "nutrition", "agriculture"],
        "Improved Transportation": ["transportation", "infrastructure", "mobility"],
        "Affordable & Clean Energy": ["clean energy", "renewable energy", "energy access"],
        "Health Care": ["health care", "medical services", "public health"],
        "Social Services": ["social services", "welfare", "community support"],
        "Education": ["education", "learning", "schools"],
        "Bureaucratic Efficiency": ["bureaucratic efficiency", "government processes", "administrative reform"],
        "Sound Fiscal Management": ["fiscal management", "budgeting", "financial planning"]
    }

    # Expand keywords with synonyms
    sdg_keywords_expanded = expand_keywords(sdg_keywords)
    rd_keywords_expanded = expand_keywords(rd_keywords)
    dost_keywords_expanded = expand_keywords(dost_keywords)
    agenda_keywords_expanded = expand_keywords(agenda_keywords)


    df['sdg'] = parallel_apply(assign_labels, df['abstract'], [sdg_keywords_expanded] * len(df))
    df['r&d'] = parallel_apply(assign_labels, df['abstract'], [rd_keywords_expanded] * len(df))
    df['dost_pillars'] = parallel_apply(assign_labels, df['abstract'], [dost_keywords_expanded] * len(df))
    df['agenda'] = parallel_apply(assign_labels, df['abstract'], [agenda_keywords_expanded] * len(df))

    # Save results
    with open('dataset/processed/v2.json', 'w') as f:
        json.dump(df.to_dict(orient='records'), f, indent=4, sort_keys=False)


