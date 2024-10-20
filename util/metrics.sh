#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_folder> <output_base_folder>"
    exit 1
fi

# Input arguments
input_folder="$1"
output_base_folder="$2"

# Check if the input folder exists
if [ ! -d "$input_folder" ]; then
    echo "Error: Input folder '$input_folder' does not exist!"
    exit 1
fi

# Create the output folder based on the input folder name
output_folder="${output_base_folder}/$(basename "$input_folder")"
mkdir -p "$output_folder"

# Loop through each CSV file in the input folder
for csv_file in "$input_folder"/*.csv; do
    if [ -f "$csv_file" ]; then
        # Get the base name of the CSV file without the extension
        base_name=$(basename "$csv_file" .csv)

        # Define the output text file
        output_file="${output_folder}/${base_name}.txt"

        # Run metrics.py on the CSV file and pipe the output to the corresponding text file
        python3 metrics.py "$csv_file" > "$output_file"

        # Print the result
        echo "Processed '$csv_file' -> '$output_file'"
    else
        echo "No CSV files found in '$input_folder'."
    fi
done
