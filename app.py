#!/usr/bin/env python
# coding: utf-8
pip install pdf2image pillow
pip install PyMuPDF

import platform
def get_poppler_path():
    if platform.system() == "Darwin":  # macOS
        if platform.machine() == "arm64":  # Apple Silicon
            return "/opt/homebrew/bin"
        else:  # Intel
            return "/usr/local/bin"
    return None  # For other operating systems, return None

POPPLER_PATH = get_poppler_path()

# # Analyze PDF pages and convert to images

import fitz  # PyMuPDF
from PIL import Image
import os

def analyze_and_convert_pdf(pdf_path, output_folder):
    
    # Open the PDF
    pdf_document = fitz.open(pdf_path)
    page_dimensions = []

    # Analyze and convert each page
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        
        # Get page dimensions
        width, height = page.rect.width, page.rect.height
        page_dimensions.append((width, height))

        # Render page to an image
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Save the image
        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        img.save(image_path)
        print(f"Saved {image_path}")

    pdf_document.close()
    return page_dimensions
    print(f"Converted {len(pages)} pages to images in {output_folder}")

# Usage
pdf_path = "resilience-chemicals-industry.pdf" #example
output_folder = "YOUR_PATH"
dimensions = analyze_and_convert_pdf(pdf_path, output_folder)

# Print dimensions of each page
for i, dim in enumerate(dimensions):
    print(f"Page {i+1} dimensions: {dim[0]}x{dim[1]} points")


# # Create grid overlay


def create_grid_overlay(width, height, num_rows, num_cols):
    
    square_width = width // num_cols
    square_height = height // num_rows
    grid = []
    
    for row in range(num_rows):
        for col in range(num_cols):
            x1 = col * square_width
            y1 = row * square_height
            x2 = x1 + square_width
            y2 = y1 + square_height
            if row == num_rows - 1:
                y2 = height
            if col == num_cols - 1:
                x2 = width
            square = {
                'row': row,
                'col': col,
                'y1': y1,
                'x1': x1,
                'y2': y2,
                'x2': x2,
                'contains_text': False,
                'contains_visual_content': False
            }
            grid.append(square)
    return grid

# Usage example (to be called for each page)
num_rows = 25
num_cols = 25

# Assuming 'dimensions' is the list returned from analyze_and_convert_pdf
for page_num, (width, height) in enumerate(dimensions):
    grid = create_grid_overlay(width, height, num_rows, num_cols)
    print(f"Created grid for page {page_num + 1}")
    print(grid)
    # Here you could save or further process the grid for each page


# # Enhanced PDF Content Detection (Text and Brightness)
# scans for text and images blocks


import fitz  # PyMuPDF
from PIL import Image
import numpy as np

def detect_content_in_grid(pdf_path, page_num, grid, brightness_threshold=20):
    # Open the PDF and get the page
    pdf_document = fitz.open(pdf_path)
    page = pdf_document[page_num]

    # Render the page as an image
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Convert to grayscale
    img_gray = img.convert('L')
    img_array = np.array(img_gray)

    # Get text blocks
    text_blocks = page.get_text("blocks")

    for square in grid:
        rect = fitz.Rect(square['x1'], square['y1'], square['x2'], square['y2'])
        square['contains_text'] = False
        square['contains_visual_content'] = False

        # Check for text
        for block in text_blocks:
            if rect.intersects(fitz.Rect(block[:4])):
                square['contains_text'] = True
                break

        # If no text, check for visual content
        if not square['contains_text']:
            # Extract the corresponding section of the grayscale image
            section = img_array[int(square['y1']):int(square['y2']), int(square['x1']):int(square['x2'])]
            
            # Calculate brightness variation
            brightness_std = np.std(section)

            # Mark as containing visual content if variation exceeds threshold
            if brightness_std > brightness_threshold:
                square['contains_visual_content'] = True

    pdf_document.close()
    return grid

# Usage example
pdf_path = "resilience-chemicals-industry.pdf"
page_num = 6  # First page
grid = create_grid_overlay(width, height, num_rows, num_cols)
grid_with_content = detect_content_in_grid(pdf_path, page_num, grid)
print(grid_with_content)


# # Visualize Grid Squares with Text

from PIL import Image, ImageDraw


def visualize_content_grid(image_path, grid, output_path, ocr_results=None, 
                           text_color=(255, 0, 0, 64), visual_content_color=(0, 0, 255, 64), 
                           ocr_text_color=(255, 255, 0, 64)):
    with Image.open(image_path) as img:
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for square in grid:
            if square['contains_text']:
                color = text_color
            elif square['contains_visual_content']:
                color = visual_content_color
            else:
                continue

            draw.rectangle(
                [square['x1'], square['y1'], square['x2'], square['y2']],
                fill=color
            )

        # Add yellow blocks for OCR-detected text
        if ocr_results:
            for result in ocr_results:
                draw.rectangle(
                    result['coordinates'],
                    fill=ocr_text_color
                )

        combined = Image.alpha_composite(img.convert('RGBA'), overlay)
        combined.save(output_path)
        print(f"Saved visualized grid to {output_path}")


image_path = "/Users/dborn/pdf2iamge/page_1.png"
output_path = "/Users/dborn/pdf2iamge/"
visualize_content_grid(image_path, grid_with_content, output_path)

# # Compare Page Structures
# 
# We process each page of the PDF as before, creating grid overlays and detecting text.
# We store the grid information for each page in all_page_grids.
# After processing all pages, we call compare_page_structures to analyze the similarity of pages based on the number of rows containing text.
# We write the results of this comparison to a text file in the output folder.

def compare_page_structures(grids):
    page_structures = []
    
    for page_num, grid in enumerate(grids):
        rows_with_text = set()
        rows_with_visual = set()
        squares_with_text = 0
        squares_with_visual = 0
        
        for square in grid:
            if square['contains_text']:
                rows_with_text.add(square['row'])
                squares_with_text += 1
            elif square['contains_visual_content']:
                rows_with_visual.add(square['row'])
                squares_with_visual += 1
        
        page_structures.append({
            'page_number': page_num + 1,
            'rows_with_text': len(rows_with_text),
            'squares_with_text': squares_with_text,
            'rows_with_visual': len(rows_with_visual),
            'squares_with_visual': squares_with_visual
        })
    
    # Compare pages
    similar_pages_rows = []
    similar_pages_squares = []
    for i in range(len(page_structures)):
        for j in range(i + 1, len(page_structures)):
            if (page_structures[i]['rows_with_text'] == page_structures[j]['rows_with_text'] and
                page_structures[i]['rows_with_visual'] == page_structures[j]['rows_with_visual']):
                similar_pages_rows.append((page_structures[i]['page_number'], page_structures[j]['page_number']))
            if (page_structures[i]['squares_with_text'] == page_structures[j]['squares_with_text'] and
                page_structures[i]['squares_with_visual'] == page_structures[j]['squares_with_visual']):
                similar_pages_squares.append((page_structures[i]['page_number'], page_structures[j]['page_number']))
    
    return page_structures, similar_pages_rows, similar_pages_squares

# Usage example
#page_structures, similar_pages_rows, similar_pages_squares = compare_page_structures(all_page_grids)


# # Inlcude OCR

pip install PyMuPDF Pillow numpy opencv-python-headless pytesseract spellchecker


# # High accuracy OCR function

import pytesseract
from PIL import Image
import cv2
import numpy as np
from autocorrect import Speller

def perform_german_ocr(image, tesseract_path):
    # Set Tesseract path
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

    # Preprocess the image
    image_array = np.array(image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(binary)

    # Perform OCR
    custom_config = r'--oem 1 --psm 1 -l eng' # uses LLM
    text = pytesseract.image_to_string(image, config=custom_config)  # Change language
    return text

# Test the function
def test_ocr(test_image_path, tesseract_path):
    image = Image.open(test_image_path)
    result = perform_german_ocr(image, tesseract_path)
    print("OCR Result:", result)

test_ocr(test_image_path, tesseract_path)
test_image_path = 'YOUR_PATH'
tesseract_path = r'/opt/homebrew/bin/tesseract'  # example path

# # Regions of ineterest on the page

def identify_regions_of_interest(grid):
    roi = []
    current_region = None
    
    for square in grid:
        if square['contains_visual_content'] and not square['contains_text']:
            if current_region is None:
                current_region = {
                    'x1': square['x1'], 'y1': square['y1'],
                    'x2': square['x2'], 'y2': square['y2']
                }
            else:
                current_region['x2'] = max(current_region['x2'], square['x2'])
                current_region['y2'] = max(current_region['y2'], square['y2'])
        else:
            if current_region is not None:
                roi.append(current_region)
                current_region = None
    
    if current_region is not None:
        roi.append(current_region)
    
    return roi


# # Process method

import fitz  # PyMuPDF
from PIL import Image
import os
import numpy as np

def process_pdf(pdf_path, output_folder, num_rows, num_cols, brightness_threshold):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pdf_document = fitz.open(pdf_path)
    all_page_grids = []

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        width, height = page.rect.width, page.rect.height

        grid = create_grid_overlay(width, height, num_rows, num_cols)
        grid_with_content = detect_content_in_grid(page, grid, brightness_threshold)

        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        original_image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        img.save(original_image_path)

        regions_of_interest = identify_regions_of_interest(grid_with_content)
        
        ocr_results = []
        for region in regions_of_interest:
            x1, y1, x2, y2 = region['x1'], region['y1'], region['x2'], region['y2']
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            
            region_image = img.crop((x1, y1, x2, y2))
            ocr_text = perform_german_ocr(region_image, tesseract_path)
            
            if ocr_text.strip():
                ocr_results.append({
                    'coordinates': (x1, y1, x2, y2),
                    'text': ocr_text
                })

        visualized_image_path = os.path.join(output_folder, f"page_{page_num + 1}_visualized.png")
        visualize_content_grid(original_image_path, grid_with_content, visualized_image_path, ocr_results)

        if ocr_results:
            ocr_file_path = os.path.join(output_folder, f"page_{page_num + 1}_ocr.txt")
            with open(ocr_file_path, 'w', encoding='utf-8') as f:
                for result in ocr_results:
                    f.write(f"Coordinates: {result['coordinates']}\n")
                    f.write(f"Text: {result['text']}\n\n")

        all_page_grids.append(grid_with_content)
        print(f"Processed page {page_num + 1}")

    pdf_document.close()
    print(f"Processed {len(pdf_document)} pages")
    
    # Compare page structures
    page_structures, similar_pages_rows, similar_pages_squares = compare_page_structures(all_page_grids)

    # Write comparison results to a file
    results_path = os.path.join(output_folder, "page_comparison_results.txt")
    with open(results_path, 'w') as f:
        f.write("Page Structure Analysis:\n\n")
        for page in page_structures:
            f.write(f"Page {page['page_number']}:\n")
            f.write(f"  - Text in {page['rows_with_text']} rows and {page['squares_with_text']} squares\n")
            f.write(f"  - Visual content in {page['rows_with_visual']} rows and {page['squares_with_visual']} squares\n")
    
    print(f"Comparison results saved to {results_path}")
        

# Usage
pdf_path = "resilience-chemicals-industry.pdf" #example
output_folder = "YOUR_OUTPUT_PATH"
num_rows = 200
num_cols = 100
brightness_threshold = 5
tesseract_path = r'/opt/homebrew/bin/tesseract'  # Update this path as per your system

process_pdf(pdf_path, output_folder, num_rows, num_cols, brightness_threshold)

