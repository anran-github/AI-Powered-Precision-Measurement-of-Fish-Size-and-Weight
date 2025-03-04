from markdown import markdown
from weasyprint import HTML

# Convert Markdown with local image support
def markdown_to_pdf(input_md_path, output_pdf_path):
    # Read the Markdown file
    with open(input_md_path, "r", encoding="utf-8") as md_file:
        markdown_text = md_file.read()
    
    # Convert Markdown to HTML
    html_content = markdown(markdown_text, extensions=['extra'])
    
    # Adjust image paths to be absolute
    base_url = input_md_path.rsplit("/", 1)[0]  # Directory of the Markdown file
    HTML(string=html_content, base_url=base_url).write_pdf(output_pdf_path)
    
    print(f"PDF created successfully: {output_pdf_path}")

# Example usage
markdown_to_pdf("process.md", "output.pdf")
