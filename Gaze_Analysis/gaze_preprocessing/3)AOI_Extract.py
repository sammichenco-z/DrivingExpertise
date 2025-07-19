import os
import csv
import xml.etree.ElementTree as ET

def extract_box_info(xml_data, condition):
    root = ET.fromstring(xml_data)
    name = root.find(".//name").text
    size = root.find(".//size").text

    box_info = []

    for image in root.findall(".//image"):
        image_id = image.get("id")
        image_name = image.get("name")
        width = image.get("width")
        height = image.get("height")


        for box in image.findall(".//box"):
            label = box.get("label")
            source = box.get("source")
            occluded = box.get("occluded")
            xtl = box.get("xtl")
            ytl = box.get("ytl")
            xbr = box.get("xbr")
            ybr = box.get("ybr")

            box_info.append([name, image_id, image_name, label, source, occluded, xtl, ytl, xbr, ybr, width, height,
                             size, condition])
    if box_info == []:
        print("EMPTY　FILE ", name)
    return box_info

def process_xml_files(root_folder):
    all_box_info = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        parts = dirpath.split(os.path.sep)
        condition_index = parts.index('EEG_DATA') + 2
        if condition_index < len(parts):
            condition = parts[condition_index]
        else:
            condition = 'Unknown'

        for filename in filenames:
            if filename.lower().endswith(".xml"):
                xml_path = os.path.join(dirpath, filename)
                with open(xml_path, "r") as xml_file:
                    xml_data = xml_file.read()
                    box_info = extract_box_info(xml_data, condition)
                    all_box_info.extend(box_info)

    return all_box_info

def save_to_csv(box_info, output_csv):
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "Image ID", "Image Name", "Label", "Source", "Occluded", "xtl", "ytl", "xbr", "ybr",
                         "width", "height", "size","condition"])
        writer.writerows(box_info)

if __name__ == "__main__":
    root_folder = "M:\\EEG_DATA\\AOIs"  # Replace with the actual path
    output_csv = "M:\\EEG_DATA\\box_info.csv"
    root_folder = "M:\\EEG_DATA\\someAOI"  # Replace with the actual path
    output_csv = "M:\\EEG_DATA\\NewCut2201.csv"

    all_box_info = process_xml_files(root_folder)
    save_to_csv(all_box_info, output_csv)

    print(f"Box information saved to {output_csv}")
