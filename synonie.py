from nltk.corpus import wordnet
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

def generate_synonyms(word):
    synonyms = set()
    
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.add(lemma.name().replace('_', ' '))
    
    return list(synonyms)

word = "happy"
synonyms = generate_synonyms(word)
print(f"Synonyms for '{word}': {synonyms}")
