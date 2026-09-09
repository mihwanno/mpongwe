import logging
import os
from tqdm import tqdm
from PyPDF2 import PdfReader, PdfWriter
import pdb



class PDFSplitter:
    def __init__(self, pdf_path: str, writing_folder: str):
        self.pdf_path = pdf_path
        self.writing_dest = writing_folder
        self.document = self._get_document()

    def _get_document(self):
        # Return the PDF document object
        return PdfReader(self.pdf_path)

    def split_pages(self):
        # Calculate total pages in the document
        total_pages = len(self.document.pages)
        # Create subfolder based on the PDF file name (without extension)
        # write_subfolder = os.path.splitext(os.path.basename(self.pdf_path))[0]
        # output_folder = os.path.join(self.writing_dest, write_subfolder)
        os.makedirs(self.writing_dest, exist_ok=True)

        # Iterate through each page and write it to a separate PDF
        for page_num in tqdm(range(total_pages), desc="Splitting PDF"):
            current_page = self.document.pages[page_num]
            writer = PdfWriter()
            writer.add_page(current_page)
            output_path = os.path.join(self.writing_dest, f"page-{page_num + 1}.pdf")
            with open(output_path, "wb") as out_file:
                writer.write(out_file)

        print(f"PDF split into {total_pages} pages and saved to '{self.writing_dest}'")

