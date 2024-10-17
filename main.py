import sys
import argparse
from loguru import logger
from bs4 import BeautifulSoup
import pandas as pd
import regex as re
import json

## Logger configuration
logger.remove()
logger.add(sys.stderr, 
           level="DEBUG",
           format="{time:HH:mm:ss} | {level} | {message}")

## Convert XML soup to JSON format
def xml_to_json(element):
    
    """
    Recursively parses XML soup, returning as JSON format 
    """
    
    if isinstance(element, str):
        return element
    
    if not element.contents:
        return element.string
    
    result = {}
    
    for child in element.children:
        
        if isinstance(child, str):
            continue
        
        if child.name not in result:
            result[child.name] = xml_to_json(child)
            
        else:
            if not isinstance(result[child.name], list):
                result[child.name] = [result[child.name]]
            result[child.name].append(xml_to_json(child))
            
    ### Directly capture text nodes without 'text' key
    if element.string and element.string.strip():
        return element.string.strip()
    
    return result


def main():
    
    """
    Accepts an XML of US OFAC sanctions information, returning a csv of relationship nodes and edges 
    """
    
    parser = argparse.ArgumentParser(description = "Process an XML file and extract relationship data.")
    
    parser.add_argument("-i",
                        "--input_file", 
                        help = "Path to the input XML file.")
    
    parser.add_argument("-o", 
                        "--output_file", 
                        help = "Path to the output file (default: output.csv or output.json)")
    
    parser.add_argument("-f",
                        "--function",
                        default = "relextract",
                        help = "Functionality, either 'relextract', 'vault', or 'json'")
    
    args = parser.parse_args()
    
    ## Load XML Sanctions Data 
    try:
        with open(args.input_file, "rb") as file:
            xml_data = file.read()
            logger.debug(f"{args.input_file} loaded")

    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input_file}")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        
        
    ## Convert XML to JSON format, isolate entity data 
    soup = BeautifulSoup(xml_data, features='xml')
    
    entity_json = xml_to_json(soup)
    entity_data = entity_json['sanctionsData']["entities"]["entity"]
    entity_data = [entity for entity in entity_data if entity["generalInfo"]["entityType"] in ["Individual", "Entity"]]
    logger.info(f"Entities found: {len(entity_data)}")
    
    
    
    ## Execute desired function
    ### Relationship extraction 
    if args.function == "relextract":
        
        import rel_extractor as rel
        logger.info("Relationship extraction selected.")
        
        #### Determine output path
        if args.output_file:
            output_file = args.output_file
        else:
            output_file = "rel_output.csv"
        
        #### Attempt to extract relationships
        try:
            df = rel.rel_extractor_main(entity_data)
            df.to_csv(output_file, index = False)
            logger.info(f"Relationship data saved as csv file: {output_file}")
        except:
            logger.error("Unable to extract relationships.")



    
    ### Entity data saved as JSON
    elif args.function == "json":
        
        import sanctions_json as sj
        logger.info("JSON conversion selected.")
        
        #### Determine output file path
        if args.output_file:
            output_file = args.output_file
        else:
            output_file = "output.json"
        
        #### Attempt to convert sanctions data to JSON file 
        try:
            entity_json = sj.json_gen(entity_data)
            with open(output_file, 'w') as output_file:
                json.dump(entity_json, output_file, indent=4)
            logger.info(f"JSON data saved as: {output_file}.")
        except:
            logger.error("Unable to convert provided data to JSON.")
            

            
            
    ## Or return DF if desired  
    # return df
    
if __name__ == "__main__":
    main()