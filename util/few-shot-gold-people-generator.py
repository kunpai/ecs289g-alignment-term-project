import random
import pycountry
import pycountry_convert as pc

# List of official United Nations member country ISO-3166 alpha-2 codes
UN_COUNTRIES = {country.alpha_2 for country in pycountry.countries}

# Function to map country to continent
def get_continent(country_alpha2):
    try:
        continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        continents = {
            "AF": "Africa",
            "AS": "Asia",
            "EU": "Europe",
            "NA": "North America",
            "SA": "South America",
            "OC": "Oceania"
        }
        return continents[continent_code]
    except KeyError:
        return None

# Create a dictionary to group official UN countries by continent
countries_by_continent = {
    "Africa": [],
    "Asia": [],
    "Europe": [],
    "North America": [],
    "South America": [],
    "Oceania": []
}

# Populate the dictionary with only UN member countries from pycountry
for country in pycountry.countries:
    if country.alpha_2 in UN_COUNTRIES:  # Ensure it’s an official UN country
        continent = get_continent(country.alpha_2)
        if continent:
            countries_by_continent[continent].append(country.name)

# Select a unique country from each continent and assign random genders
continents = list(countries_by_continent.keys())
random.shuffle(continents)

genders = ["Male"] * 3 + ["Female"] * 3
random.shuffle(genders)
selection = []

for i, continent in enumerate(continents):
    if i >= 6:  # We only need 6 continents
        break
    country = random.choice(countries_by_continent[continent])
    gender = genders[i]
    selection.append((gender, country, continent))

# Print the result
for gender, country, continent in selection:
    print(f"Gender: {gender}, Country: {country}")
