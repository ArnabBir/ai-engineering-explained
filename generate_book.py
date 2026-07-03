import os
import subprocess

# --- CONFIGURATION ---
# Define the order of your chapters. If empty, it will sort them alphabetically.
# Example: ["introduction.md", "chapter1.md", "chapter2.md", "conclusion.md"]
CHAPTER_ORDER = [] 

OUTPUT_PDF = "my_printed_book.pdf"
BOOK_TITLE = "My Awesome Book"
AUTHOR = "Your Name"
# ---------------------

def get_markdown_files():
    """Gets and sorts the markdown files based on config or alphabet."""
    if CHAPTER_ORDER:
        # Filter to ensure the files actually exist
        missing = [f for f in CHAPTER_ORDER if not os.path.exists(f)]
        if missing:
            print(f"⚠️ Warning: Configured files not found: {missing}")
        return [f for f in CHAPTER_ORDER if os.path.exists(f)]
    
    # Otherwise, grab all .md files and sort them alphabetically
    files = [f for f in os.listdir('.') if f.endswith('.md') and f != 'combined_book.md']
    return sorted(files)

def combine_markdowns(files):
    """Combines files into a single master markdown with explicit page breaks."""
    combined_filename = "combined_book.md"
    
    # YAML metadata block for Pandoc to style the book
    yaml_header = f"""---
title: "{BOOK_TITLE}"
author: "{AUTHOR}"
geometry: "margin=1in"
output: pdf_document
---

"""
    
    with open(combined_filename, "w", encoding="utf-8") as outfile:
        outfile.write(yaml_header)
        
        for i, file_name in enumerate(files):
            print(f"📖 Processing: {file_name}")
            with open(file_name, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
                
            # Inject a LaTeX page break after every chapter except the last one
            if i < len(files) - 1:
                outfile.write("\n\n\\newpage\n\n")
                
    return combined_filename

def compile_to_pdf(combined_md, output_pdf):
    """Calls Pandoc to convert the combined markdown into a print-ready PDF."""
    print("🎨 Compiling PDF with Pandoc (this might take a moment)...")
    
    cmd = [
        "pandoc",
        combined_md,
        "-o", output_pdf,
        "--toc",                  # Generates an automatic Table of Contents
        "--toc-depth=2",          # Includes headers up to H2 in the TOC
        "--pdf-engine=xelatex",   # Handles fonts and formatting cleanly
        "-V", "documentclass=report" # 'report' or 'book' style is ideal for printing
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"🎉 Success! Your book is ready at: {output_pdf}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during PDF compilation: {e}")
    finally:
        # Clean up the temporary combined markdown file
        if os.path.exists(combined_md):
            os.remove(combined_md)

if __name__ == "__main__":
    md_files = get_markdown_files()
    
    if not md_files:
        print("❌ No markdown files found in the current directory.")
    else:
        print(f"📚 Found {len(md_files)} markdown files.")
        temp_md = combine_markdowns(md_files)
        compile_to_pdf(temp_md, OUTPUT_PDF)

