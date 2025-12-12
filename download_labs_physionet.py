"""
Download Labs.csv from PhysioNet VitalDB dataset
"""

import requests
import os
from tqdm import tqdm

# Disable SSL warnings (only for this specific case)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_labs_csv(output_path):
    """Download Labs.csv from PhysioNet"""

    # PhysioNet requires authentication or the files are available without auth
    # Try different URL patterns
    urls = [
        "https://physionet.org/static/published-projects/vitaldb/vitaldb-a-high-fidelity-multi-parameter-vital-signs-database-in-surgical-patients-1.0.0/Labs.csv",
        "https://physionet.org/files/vitaldb/1.0.0/Labs.csv",
        "https://physionet.org/content/vitaldb/1.0.0/files/Labs.csv",
    ]

    print("=" * 80)
    print("DOWNLOADING LABS.CSV FROM PHYSIONET")
    print("=" * 80)

    for i, url in enumerate(urls, 1):
        print(f"\n[Attempt {i}/{len(urls)}] Trying: {url}")

        try:
            # Make request
            response = requests.get(url, stream=True, verify=False, timeout=30)

            if response.status_code == 200:
                # Get file size
                total_size = int(response.headers.get('content-length', 0))

                # Create output directory
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Download with progress bar
                print(f"[OK] Found! Downloading...")
                print(f"Size: {total_size / (1024*1024):.2f} MB")

                with open(output_path, 'wb') as f:
                    if total_size > 0:
                        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    else:
                        f.write(response.content)

                print(f"\n[SUCCESS] Downloaded to: {output_path}")
                return True

            else:
                print(f"[FAILED] Status code: {response.status_code}")

        except Exception as e:
            print(f"[ERROR] {e}")

    print("\n" + "=" * 80)
    print("[FAILED] Could not download from any source")
    print("=" * 80)
    print("\nMANUAL DOWNLOAD REQUIRED:")
    print("1. Go to: https://physionet.org/content/vitaldb/1.0.0/")
    print("2. Click on 'Files' tab")
    print("3. Download 'Labs.csv'")
    print(f"4. Save to: {output_path}")
    print("\nNote: PhysioNet may require you to:")
    print("  - Create a free account")
    print("  - Accept the data use agreement")
    print("  - Sign the data use agreement for the VitalDB dataset")
    return False


if __name__ == '__main__':
    output_path = r"C:\IITM\vitalDB\data\labs\Labs.csv"
    success = download_labs_csv(output_path)

    if success:
        # Analyze the file
        import pandas as pd
        print("\n" + "=" * 80)
        print("ANALYZING LABS.CSV")
        print("=" * 80)

        labs_df = pd.read_csv(output_path)
        print(f"\nTotal records: {len(labs_df):,}")
        print(f"Columns: {labs_df.columns.tolist()}")
        print(f"\nFirst 10 rows:")
        print(labs_df.head(10))

        # Look for glucose
        if 'test' in labs_df.columns:
            glucose_records = labs_df[labs_df['test'].str.contains('GLU|glucose', case=False, na=False)]
            print(f"\n[OK] Found {len(glucose_records):,} glucose records!")
        elif 'name' in labs_df.columns:
            glucose_records = labs_df[labs_df['name'].str.contains('GLU|glucose', case=False, na=False)]
            print(f"\n[OK] Found {len(glucose_records):,} glucose records!")
