import xml.etree.ElementTree as ET
import pandas as pd


def parse_xml(xml_file):
    # Parse XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Initialize lists to store data
    data = []

    # Process each track in the XML
    for track in root.findall('.//track'):
        # Process each box in the track
        for box in track.findall('box'):
            # Only include boxes where outside="0"
            if box.get('outside') == '0':
                # Extract the data
                row = {
                    'Name': 'NewCut_2201',
                    'Label': 'Occlusion-OR',
                    'xtl': float(box.get('xtl')),
                    'ytl': float(box.get('ytl')),
                    'xbr': float(box.get('xbr')),
                    'ybr': float(box.get('ybr')),
                    'Condition': 'OcclusionOnly',
                    'Image ID': int(box.get('frame'))
                }
                data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort by Image ID
    df = df.sort_values('Image ID')

    # Remove duplicate rows
    df = df.drop_duplicates()

    return df



# Use the function
df = parse_xml('M:\\EEG_DATA\\someAOI\\annotations.xml')

# Save to CSV
df.to_csv('M:\\EEG_DATA\\someAOI\\output.csv', index=False)

# Display first few rows
print(df)