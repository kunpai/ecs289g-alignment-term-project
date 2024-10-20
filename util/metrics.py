import pandas as pd
import argparse

def calculate_demographic_parity_gender(df, gender):
    # Calculate demographic parity for a specific inferred gender
    total_count = len(df)
    favorable_count = len(df[df['Inferred_Gender'] == gender])
    demographic_parity = favorable_count / total_count if total_count != 0 else 0
    return demographic_parity

def calculate_demographic_parity_nationality(df, nationality):
    # Calculate demographic parity for a specific inferred nationality
    total_count = len(df)
    favorable_count = len(df[df['Inferred_Nationality'] == nationality])
    demographic_parity = favorable_count / total_count if total_count != 0 else 0
    return demographic_parity

def calculate_disparate_impact(df, nationality1, nationality2, gender):
    # Calculate the selection rates for each nationality based on inferred gender
    nationality1_count = len(df[df['Inferred_Nationality'] == nationality1])
    nationality1_favorable = len(df[(df['Inferred_Nationality'] == nationality1) & (df['Inferred_Gender'] == gender)])
    nationality1_selection_rate = nationality1_favorable / nationality1_count if nationality1_count != 0 else 0

    nationality2_count = len(df[df['Inferred_Nationality'] == nationality2])
    nationality2_favorable = len(df[(df['Inferred_Nationality'] == nationality2) & (df['Inferred_Gender'] == gender)])
    nationality2_selection_rate = nationality2_favorable / nationality2_count if nationality2_count != 0 else 0

    # Disparate Impact ratio
    disparate_impact = nationality1_selection_rate / nationality2_selection_rate if nationality2_selection_rate != 0 else None
    return disparate_impact

def main(csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file, delimiter=';')

    # Get unique inferred genders and nationalities from the CSV
    unique_genders = df['Inferred_Gender'].unique()
    unique_nationalities = df['Inferred_Nationality'].unique()

    # Calculate Demographic Parity for each inferred gender
    print("Demographic Parity by Gender:")
    for gender in unique_genders:
        dp = calculate_demographic_parity_gender(df, gender)
        print(f"  - {gender}: {dp:.2f}")

    print("*" * 50)

    print("\nDemographic Parity by Nationality:")
    # Calculate Demographic Parity for each inferred nationality
    for nationality in unique_nationalities:
        dp = calculate_demographic_parity_nationality(df, nationality)
        print(f"  - {nationality}: {dp:.2f}")

    print("*" * 50)

    # Calculate Disparate Impact between each pair of inferred nationalities for each inferred gender
    print("\nDisparate Impact:")
    for gender in unique_genders:
        print(f"  For gender: {gender}")
        for i in range(len(unique_nationalities)):
            for j in range(i + 1, len(unique_nationalities)):
                nationality1 = unique_nationalities[i]
                nationality2 = unique_nationalities[j]
                di = calculate_disparate_impact(df, nationality1, nationality2, gender)

                if di is not None:
                    print(f"    - {nationality1} vs {nationality2}: {di:.2f}")
                else:
                    print(f"    - {nationality1} vs {nationality2}: Calculation not possible (division by zero)")

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Calculate Demographic Parity and Disparate Impact from CSV data.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')

    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.csv_file)
