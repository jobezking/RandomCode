#!/usr/bin/env python3
import pandas as pd
from faker import Faker
import os

def create_large_csv(file_name, target_gb):
    """
    Creates a large CSV file with a specified target size in gigabytes.

    Args:
        file_name (str): The name of the CSV file to create.
        target_gb (float): The target size of the file in gigabytes.
    """
    fake = Faker()
    # Estimate the number of rows needed to reach the target size.
    # This is an approximation. A typical row with a few columns might be
    # around 100-200 bytes, so 1GB is roughly 5-10 million rows.
    # We'll use a conservative estimate to ensure the file is large enough.
    bytes_per_row = 150  # Estimated bytes per row.
    target_bytes = target_gb * 1024**3
    num_rows_per_chunk = 1000000
    
    # Calculate the number of chunks needed.
    num_chunks = int(target_bytes / (num_rows_per_chunk * bytes_per_row)) + 1
    
    print(f"Starting to create {target_gb} GB CSV file: {file_name}")
    print(f"Estimated number of chunks: {num_chunks}")

    with open(file_name, 'w') as f:
        # Write the header row
        header = ['id', 'name', 'address', 'email', 'phone_number', 'company', 'job_title', 'birthdate']
        f.write(','.join(header) + '\n')
    
        for i in range(num_chunks):
            # Generate a chunk of data
            data = []
            for _ in range(num_rows_per_chunk):
                row = [
                    str(fake.uuid4()),
                    fake.name(),
                    fake.address().replace('\n', ', '),
                    fake.email(),
                    fake.phone_number(),
                    fake.company(),
                    fake.job(),
                    str(fake.date_of_birth())
                ]
                data.append(row)
                
            # Create a DataFrame for the chunk and append to the CSV
            df_chunk = pd.DataFrame(data, columns=header)
            df_chunk.to_csv(file_name, mode='a', header=False, index=False)
            
            # Print progress and current file size
            current_size_gb = os.path.getsize(file_name) / 1024**3
            print(f"Chunk {i+1}/{num_chunks} processed. Current file size: {current_size_gb:.2f} GB")
            if current_size_gb >= target_gb:
                print(f"Target size of {target_gb} GB reached. Stopping.")
                break
                
    print(f"CSV file '{file_name}' created successfully. Final size: {os.path.getsize(file_name) / 1024**3:.2f} GB")
    
if __name__ == '__main__':
    create_large_csv('middle_stress_test.csv', 18)
