
import re

def clean_text(text):
    # Remove headers, footers, page numbers, and other noise
    # This is a basic cleaning, more sophisticated methods might be needed
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove excessive newlines
    text = re.sub(r'\f', '', text)  # Remove form feed characters
    text = re.sub(r'Page \d+ of \d+', '', text)  # Remove page numbers (example pattern)
    text = re.sub(r'\b\d+\b', '', text)  # Remove standalone numbers that might be page numbers
    text = re.sub(r'\s{2,}', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

def segment_text(text):
    # Simple segmentation based on common financial report headings
    sections = {
        "Income Statement": [],
        "Balance Sheet": [],
        "Cash Flow Statement": [],
        "Notes to Financial Statements": [],
        "Directors Report": [],
        "Auditors Report": [],
        "Other": []
    }
    
    current_section = "Other"
    lines = text.split('\n')
    
    for line in lines:
        if re.search(r'income statement', line, re.IGNORECASE):
            current_section = "Income Statement"
        elif re.search(r'balance sheet', line, re.IGNORECASE):
            current_section = "Balance Sheet"
        elif re.search(r'cash flow statement', line, re.IGNORECASE):
            current_section = "Cash Flow Statement"
        elif re.search(r'notes to financial statements', line, re.IGNORECASE):
            current_section = "Notes to Financial Statements"
        elif re.search(r'directors report', line, re.IGNORECASE):
            current_section = "Directors Report"
        elif re.search(r'auditors report', line, re.IGNORECASE):
            current_section = "Auditors Report"
        
        sections[current_section].append(line)
            
    # Join lines back into sections
    for key, value in sections.items():
        sections[key] = '\n'.join(value)
        
    return sections

if __name__ == "__main__":
    for year in ["2022-23", "2023-24"]:
        input_file = f"MM-Annual-Report-{year}.txt"
        output_cleaned_file = f"MM-Annual-Report-{year}_cleaned.txt"
        
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        cleaned_text = clean_text(raw_text)
        
        with open(output_cleaned_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
            
        print(f"Cleaned text saved to {output_cleaned_file}")
        
        # Segmentation and saving each section to a separate file
        segmented_content = segment_text(cleaned_text)
        for section_name, content in segmented_content.items():
            if content.strip(): # Only save non-empty sections
                section_filename = f"MM-Annual-Report-{year}_{section_name.replace(' ', '_')}.txt"
                with open(section_filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Segmented section '{section_name}' saved to {section_filename}")



