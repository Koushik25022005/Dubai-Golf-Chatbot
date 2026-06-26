import os
import json
import re
from bs4 import BeautifulSoup

INPUT_FILE = "raw_data.jsonl"
OUTPUT_FILE = "cleaned_data.jsonl"

def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style", "header", "footer", "nav"]):
        script.extract()
        
    # Extract text
    text = soup.get_text(separator=' ')
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def clean_data():
    input_path = os.path.join(os.path.dirname(__file__), INPUT_FILE)
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILE)
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run scraper first.")
        return
        
    print(f"Processing raw data from {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
         
        for line in infile:
            data = json.loads(line)
            url = data['url']
            html = data['html']
            
            clean_text = clean_html(html)
            
            # Create overlapping chunks
            chunks = chunk_text(clean_text)
            
            for idx, chunk in enumerate(chunks):
                clean_data = {
                    "url": url,
                    "chunk_id": f"{url}#{idx}",
                    "text": chunk
                }
                outfile.write(json.dumps(clean_data) + "\n")
                
    print(f"Data cleaning completed. Cleaned chunks saved to {output_path}")

if __name__ == "__main__":
    clean_data()
