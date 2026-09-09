import os
from services.pdf_processor import PDFSplitter

default_pdf_path = os.path.join(os.path.dirname(__file__), "../../data/raw/Raponda-Walker_1995_Dictionnaire_Mpongwe-Francais.pdf")
output_folder = os.path.join(os.path.dirname(__file__), "../../data/intermediate/pages/")


# Initialize the PDFSplitter with the default path
splitter = PDFSplitter(default_pdf_path, output_folder)

# Split the PDF into individual pages
splitter.split_pages()