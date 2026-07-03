import zipfile
import os
import re

def update_resume_docx(docx_path, github_link):
    if not os.path.exists(docx_path):
        print(f"Warning: File not found at {docx_path}")
        return False
        
    temp_zip_path = docx_path + ".temp"
    
    try:
        # Open the original docx as a zip and copy files, modifying word/document.xml
        with zipfile.ZipFile(docx_path, 'r') as zin:
            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zout:
                for item in zin.infolist():
                    data = zin.read(item.filename)
                    if item.filename == 'word/document.xml':
                        # Decode the XML file
                        xml_content = data.decode('utf-8')
                        
                        # Replace the LINKEDLN placeholder with LinkedIn + GitHub Link
                        # Checking both spellings in case
                        if '|LINKEDLN|' in xml_content:
                            new_text = f"|LINKEDLN|   |   GitHub: {github_link}"
                            xml_content = xml_content.replace('|LINKEDLN|', new_text)
                            print(f"Replaced '|LINKEDLN|' in {docx_path}")
                        elif '|LINKEDIN|' in xml_content:
                            new_text = f"|LINKEDIN|   |   GitHub: {github_link}"
                            xml_content = xml_content.replace('|LINKEDIN|', new_text)
                            print(f"Replaced '|LINKEDIN|' in {docx_path}")
                        else:
                            # Fallback: search for email and append it after email contact block
                            email_pattern = r"(sailokesh2706g\s*@gmail\.com)"
                            if re.search(email_pattern, xml_content):
                                xml_content = re.sub(email_pattern, r"\1   |   GitHub: " + github_link, xml_content)
                                print(f"Appended GitHub link next to email in {docx_path}")
                            else:
                                print(f"Placeholder not found in {docx_path}, appending to end of doc")
                        
                        data = xml_content.encode('utf-8')
                    zout.writestr(item, data)
                    
        # Replace original with the updated file
        os.replace(temp_zip_path, docx_path)
        print(f"Successfully updated resume at: {docx_path}")
        return True
    except Exception as e:
        print(f"Failed to update docx at {docx_path}: {e}")
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python update_resume.py <github_link>")
        sys.exit(1)
        
    github_link = sys.argv[1]
    
    resumes = [
        r"C:\Users\sailo\OneDrive\RESUME SAI.docx",
        r"C:\Users\sailo\OneDrive\Documents\RESUME SAI.docx"
    ]
    
    for r in resumes:
        update_resume_docx(r, github_link)
