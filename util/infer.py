import spacy
import pycountry
import nltk
import gender_guesser.detector as gender
from nltk.corpus import names
from nltk.corpus import wordnet as wn

# Download necessary NLTK data
nltk.download('names')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load Spacy's pre-trained model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Initialize the gender detector
detector = gender.Detector()

# List of gender-specific pronouns
MALE_PRONOUNS = ["he", "him", "his"]
FEMALE_PRONOUNS = ["she", "her", "hers"]

def get_country_from_nationality(nationality):
    """
    Convert nationality adjective to country name using multiple approaches
    Example: 'Greek' -> 'Greece', 'French' -> 'France'
    """
    nationality = nationality.lower().strip()

    # Common direct mappings that might be missed by other methods
    direct_mapping = {
        'french': 'France',
        'greek': 'Greece',
        'dutch': 'Netherlands',
        'danish': 'Denmark',
        'finnish': 'Finland',
        'swedish': 'Sweden',
        'english': 'United Kingdom',
        'british': 'United Kingdom',
        'welsh': 'United Kingdom',
        'scotch': 'United Kingdom',
        'scottish': 'United Kingdom',
        'irish': 'Ireland',
        'spanish': 'Spain',
        'portuguese': 'Portugal',
        'german': 'Germany',
        'italian': 'Italy',
        'russian': 'Russia',
        'chinese': 'China',
        'japanese': 'Japan',
        'vietnamese': 'Vietnam',
        'korean': 'Korea',
        'iranian': 'Iran',
        'iraqi': 'Iraq',
        'saudi': 'Saudi Arabia',
        'egyptian': 'Egypt',
        'australian': 'Australia',
        'canadian': 'Canada',
        'brazilian': 'Brazil',
        'mexican': 'Mexico',
        'indian': 'India',
        'pakistani': 'Pakistan',
        'turkish': 'Turkey',
        'indonesian': 'Indonesia',
        'nigerian': 'Nigeria',
        'south african': 'South Africa',
        'kenyan': 'Kenya',
        'ugandan': 'Uganda',
        'ethiopian': 'Ethiopia',
        'moroccan': 'Morocco',
        'israeli' : 'Israel',
        'palenstinian': 'Palestine',
        'syrian': 'Syria',
        'lebanese': 'Lebanon',
        'jordanian': 'Jordan',
        'afghan': 'Afghanistan',
        'kazakh': 'Kazakhstan',
        'uzbek': 'Uzbekistan',
        'tajik': 'Tajikistan',
        'turkmen': 'Turkmenistan',
        'mongolian': 'Mongolia',
        'thai': 'Thailand',
        'filipino': 'Philippines',
        'malaysian': 'Malaysia',
        'singaporean': 'Singapore',
        'indonesian': 'Indonesia',
        'vietnamese': 'Vietnam',
        'laotian': 'Laos',
        'cambodian': 'Cambodia',
        'myanmar': 'Myanmar',
    }

    # Check direct mapping first
    if nationality in direct_mapping:
        return direct_mapping[nationality]

    # Try WordNet lookup with improved synset traversal
    for synset in wn.synsets(nationality):
        # Check definition for country-related terms
        if any(word in synset.definition().lower() for word in ['citizen', 'national', 'people', 'language', 'country']):
            # Look for related noun forms
            for lemma in synset.lemmas():
                related_forms = lemma.derivationally_related_forms()
                for related_form in related_forms:
                    related_synset = related_form.synset()
                    if related_synset.pos() == wn.NOUN:
                        # Verify it's a country by checking hypernyms
                        for hypernym in related_synset.hypernyms():
                            if any(word in hypernym.lemma_names() for word in ['country', 'nation', 'state']):
                                return related_form.name().replace('_', ' ').title()

    # Try suffix replacement rules if WordNet lookup fails
    suffix_rules = [
        ('ish', ''),      # British -> Britain
        ('ese', ''),      # Japanese -> Japan
        ('ian', 'ia'),    # Russian -> Russia
        ('i', 'ia'),      # Iraqi -> Iraq
        ('ic', 'ia'),     # Germanic -> Germania
        ('ean', 'ea'),    # European -> Europe
        ('an', 'a'),      # American -> America
    ]

    for suffix, replacement in suffix_rules:
        if nationality.endswith(suffix):
            potential_country = nationality[:-len(suffix)] + replacement
            # Validate with pycountry
            try:
                matches = pycountry.countries.search_fuzzy(potential_country)
                if matches:
                    return matches[0].name
            except LookupError:
                continue

    # Try direct pycountry lookup as last resort
    try:
        matches = pycountry.countries.search_fuzzy(nationality)
        if matches:
            return matches[0].name
    except LookupError:
        pass

    # Return original if all methods fail
    return nationality.title()

def infer_gender_from_name(name):
    first_name = name.split()[0]
    gender_prediction = detector.get_gender(first_name)
    return gender_prediction

def infer_gender_from_pronouns(doc):
    male_count = sum([1 for token in doc if token.text.lower() in MALE_PRONOUNS])
    female_count = sum([1 for token in doc if token.text.lower() in FEMALE_PRONOUNS])
    if male_count > female_count:
        return "male"
    elif female_count > male_count:
        return "female"
    else:
        return "unknown"

def infer_nationality(doc):
    # First check for explicit nationality mentions (NORP entities)
    for ent in doc.ents:
        if ent.label_ == "NORP":
            country = get_country_from_nationality(ent.text)
            if country:
                return country

    # Check for nationality adjectives
    for token in doc:
        if token.pos_ == "ADJ" and token.ent_type_ == "NORP":
            country = get_country_from_nationality(token.text)
            if country:
                return country

    # Check for direct country mentions
    for ent in doc.ents:
        if ent.label_ == "GPE":
            try:
                country = pycountry.countries.search_fuzzy(ent.text)
                if country:
                    return country[0].name
            except LookupError:
                continue

    return None

def extract_info(paragraph):
    doc = nlp(paragraph)
    person_name = None
    country = None
    inferred_gender = "unknown"

    # Extract persons and locations from the text
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            person_name = ent.text

    # Infer nationality
    country = infer_nationality(doc)

    # Infer gender from pronouns in the paragraph
    pronoun_gender = infer_gender_from_pronouns(doc)

    # If no pronouns are found, infer gender from the person's name
    if pronoun_gender == "unknown" and person_name:
        inferred_gender = infer_gender_from_name(person_name)
    else:
        inferred_gender = pronoun_gender

    return person_name, inferred_gender, country


if __name__ == "__main__":
    # Test the functionality
    test_paragraphs = [
        """Maria Papadopoulos, a brilliant Greek scientist, has contributed significantly to modern physics.
        She has worked on numerous projects in Greece and internationally.""",

        """Jean Dupont, a French researcher, published groundbreaking work in mathematics.
        His theories have influenced scholars worldwide.""",

        """Yuki Tanaka is a Japanese author known for her novels about modern society.
        She currently resides in Tokyo.""",

        """Carlos Rodriguez, a Spanish engineer, developed innovative solar technologies.
        His work has been implemented across Europe.""",

        """Anna Schmidt is a German physicist working on quantum mechanics.
        She leads a research team in Berlin.""",

        """Alexis Smith is a British artist known for his abstract paintings.
        He has exhibited his work in galleries around the world.""",

        """Ling Wong is a Chinese entrepreneur who founded a successful tech startup.
        She is based in Shanghai.""",

        """Ahmed Hassan, an Egyptian architect, designed iconic buildings in Cairo.
        His projects have received international acclaim.""",

        """Sofia Costa is from Palau and works as a marine biologist.
        She studies coral reefs in the Pacific Ocean."""
    ]

    for i, para in enumerate(test_paragraphs, 1):
        person_name, inferred_gender, inferred_country = extract_info(para)
        print(f"\nTest case {i}:")
        print(f"Person: {person_name}")
        print(f"Inferred Gender: {inferred_gender}")
        print(f"Inferred Country: {inferred_country}")