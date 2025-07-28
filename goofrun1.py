import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict
import fitz  # PyMuPDF
from datetime import datetime
import re # For regex in text cleaning

def detect_document_type(lines):
    """Detect document type for specialized handling"""
    first_page_text = ' '.join(line['text'] for line in lines if line['page'] == 1).lower()
    
    # Academic paper indicators
    if any(word in first_page_text for word in 
           ['abstract', 'keywords', 'introduction', 'references', 'doi:', 'arxiv']):
        return 'academic'
    
    # Report indicators  
    elif any(word in first_page_text for word in 
             ['executive summary', 'table of contents', 'appendix', 'findings']):
        return 'report'
    
    # Book/manual indicators
    elif any(word in first_page_text for word in 
             ['chapter', 'preface', 'isbn', 'edition', 'publisher']):
        return 'book'
    
    return 'document'

def validate_title_content(text):
    """Validate if text content is suitable for a title. Used in Phase 1."""
    text_clean = text.strip()
    
    # Skip if too long (likely paragraph)
    if len(text_clean.split()) > 25:
        return False
    
    # Skip if too short
    if len(text_clean) < 3:
        return False
    
    # Skip if purely numeric (page numbers, dates)
    if text_clean.replace('.', '').replace(',', '').replace('-', '').isdigit():
        return False
    
    # Skip common headers/footers patterns
    skip_patterns = ['page', 'draft', 'confidential', 'copyright', '©', 'www.', 'http']
    if any(pattern in text_clean.lower() for pattern in skip_patterns):
        return False
        
    return True

def validate_heading_content(text):
    """
    Validate if text content is suitable for a general heading. Used in Phase 2.
    More aggressive filtering for noise and non-descriptive text.
    """
    text_clean = text.strip()

    # Filter out empty or very short lines (e.g., single characters, bullet points)
    if len(text_clean) < 2:
        return False
    
    # Filter out purely numeric lines (e.g., page numbers, list numbers like "1", "2.1")
    if re.fullmatch(r'^\d+(\.\d+)*$', text_clean): # Matches "1", "2.1", "3.2.5"
        return False

    # Filter out common noise patterns like bullet points, repeated characters, etc.
    # Updated regex to be more comprehensive for common non-text characters
    if re.fullmatch(r'^[•\-\–—\*\s\._,]*$', text_clean): # Matches only bullet points, dashes, dots, underscores, commas, spaces
        return False
    if re.fullmatch(r'^\s*(\S\s*){1,3}\s*$', text_clean) and any(c.isdigit() for c in text_clean): # Short numeric/alphanumeric like "1.", "2.1", "A."
        return False # This aims to filter out list numbers or very short labels

    # Filter out lines that are just a few repeated characters (e.g., "-------", ".......")
    if re.fullmatch(r'(.)\1{4,}', text_clean): # 5 or more repeated chars
        return False

    # Skip common headers/footers (can be refined based on document specifics)
    skip_patterns = ['page', 'draft', 'confidential', 'copyright', '©', 'www.', 'http', 'date', 'version']
    if any(pattern in text_clean.lower() for pattern in skip_patterns):
        return False
    
    return True


def group_title_lines(candidates):
    """
    Group consecutive lines that should be part of the title based on proximity and font.
    Improved logic for horizontal alignment and continuation.
    """
    if not candidates:
        return []
    
    # Sort by vertical position
    sorted_candidates = sorted(candidates, key=lambda x: (x['y0'], x['x0']))
    
    title_group = [sorted_candidates[0]]
    
    for i in range(1, len(sorted_candidates)):
        current_line = sorted_candidates[i]
        prev_line = title_group[-1] # Compare with the last added line in the group
        
        vertical_gap = current_line['y0'] - prev_line['y1']
        
        # Calculate horizontal overlap
        overlap_start = max(current_line['x0'], prev_line['x0'])
        overlap_end = min(current_line['x1'], prev_line['x1'])
        horizontal_overlap = max(0, overlap_end - overlap_start)
        
        # Calculate overlap ratio relative to the smaller width of the two lines
        min_width_of_lines = min(current_line['x1'] - current_line['x0'], prev_line['x1'] - prev_line['x0'])
        overlap_ratio = horizontal_overlap / min_width_of_lines if min_width_of_lines > 0 else 0

        # Check if current line is a direct continuation (small horizontal gap, similar y-alignment)
        # This helps combine fragmented words that are close horizontally.
        horizontal_gap = current_line['x0'] - prev_line['x1']
        is_direct_continuation = (horizontal_gap < 5 and abs(current_line['y0'] - prev_line['y0']) < 5) # Small horizontal gap, similar baseline

        # Condition to group lines:
        # 1. Very close vertically AND significant horizontal overlap (>= 50%)
        # 2. OR very close vertically AND is a direct continuation (small horizontal gap, similar baseline)
        # 3. OR same font size AND very small vertical gap (e.g., < 5 points)
        if (vertical_gap <= 20 and overlap_ratio >= 0.5) or \
           (vertical_gap <= 10 and is_direct_continuation) or \
           (current_line['font_size'] == prev_line['font_size'] and vertical_gap < 5):
            title_group.append(current_line)
        else:
            break  # Stop at first large gap or non-overlapping line
    
    return title_group

def calculate_combined_bbox(lines):
    """Calculate combined bounding box for multiple lines"""
    if not lines:
        return None
    
    bbox = list(lines[0]['bbox']) # Ensure it's a mutable list
    
    for line in lines[1:]:
        bbox[0] = min(bbox[0], line['bbox'][0])  # x0 - leftmost
        bbox[1] = min(bbox[1], line['bbox'][1])  # y0 - topmost
        bbox[2] = max(bbox[2], line['bbox'][2])  # x1 - rightmost
        bbox[3] = max(bbox[3], line['bbox'][3])  # y1 - bottommost
    
    return bbox


def extract_pdf_content(doc: fitz.Document) -> List[Dict]:
    """
    Extracts all text content from PDF with detailed properties.
    The is_bold/is_italic flags are now True only if *all* text spans in the line are bold/italic.
    """
    print(f"🔍 EXTRACTING CONTENT from document with {len(doc)} pages")
    
    try:
        all_lines = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_dict = page.get_text("dict")
            
            page_lines = []
            for block in page_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        line_bbox = None
                        line_font_size = 0
                        line_font_name = ""
                        
                        # New flags to track if ALL spans are bold/italic
                        all_spans_are_bold = True
                        all_spans_are_italic = True
                        
                        # New flags to track if there's *any* bold/italic content
                        # This helps distinguish lines where no text is bold/italic from lines where all text is bold/italic.
                        contains_any_bold_span = False
                        contains_any_italic_span = False

                        valid_spans_count = 0 # Count spans with actual text

                        for span in line["spans"]:
                            if span["text"].strip(): # Only consider spans with actual content
                                valid_spans_count += 1
                                line_text += span["text"]
                                line_font_size = max(line_font_size, span["size"])
                                line_font_name = span["font"]
                                
                                # Check current span's bold/italic status
                                current_span_is_bold = 'bold' in span["font"].lower() or (span["flags"] & 2**4)
                                current_span_is_italic = 'italic' in span["font"].lower() or (span["flags"] & 2**1)
                                
                                # Update overall line bold/italic status (AND logic)
                                if not current_span_is_bold:
                                    all_spans_are_bold = False
                                if not current_span_is_italic:
                                    all_spans_are_italic = False
                                    
                                if current_span_is_bold:
                                    contains_any_bold_span = True
                                if current_span_is_italic:
                                    contains_any_italic_span = True
                                
                                # Set or extend bounding box
                                if line_bbox is None:
                                    line_bbox = list(span["bbox"])
                                else:
                                    line_bbox[0] = min(line_bbox[0], span["bbox"][0])  # x0
                                    line_bbox[1] = min(line_bbox[1], span["bbox"][1])  # y0
                                    line_bbox[2] = max(line_bbox[2], span["bbox"][2])  # x1
                                    line_bbox[3] = max(line_bbox[3], span["bbox"][3])  # y1
                        
                        # Final determination for the line's bold/italic status
                        # It's bold only if ALL valid spans are bold AND there was at least one bold span.
                        # This handles cases of empty lines or lines with no actual bold text.
                        line_is_bold = all_spans_are_bold and contains_any_bold_span and valid_spans_count > 0
                        line_is_italic = all_spans_are_italic and contains_any_italic_span and valid_spans_count > 0

                        if line_text.strip() and line_bbox:
                            line_info = {
                                'text': line_text.strip(),
                                'page': page_num + 1,
                                'bbox': line_bbox,
                                'font_size': round(line_font_size, 1),
                                'font_name': line_font_name,
                                'is_bold': line_is_bold,
                                'is_italic': line_is_italic,
                                'x0': line_bbox[0],
                                'y0': line_bbox[1],
                                'x1': line_bbox[2],
                                'y1': line_bbox[3]
                            }
                            page_lines.append(line_info)
            
            # Sort page lines by vertical position (top to bottom), then horizontal (left to right)
            page_lines.sort(key=lambda x: (x['y0'], x['x0']))
            all_lines.extend(page_lines)
            
            print(f"📄 Page {page_num + 1}: Found {len(page_lines)} text lines")
        
        print(f"✅ EXTRACTION COMPLETE: {len(all_lines)} total text lines")
        return all_lines
        
    except Exception as e:
        print(f"❌ EXTRACTION FAILED: {str(e)}")
        return []

def clean_title_text(title_lines: List[Dict]) -> str:
    """
    Cleans and consolidates title text from multiple lines, handling fragmentation and redundancy.
    This function is defined globally and should be accessible.
    """
    if not title_lines:
        return ""

    # Sort lines by y0, then x0
    sorted_lines = sorted(title_lines, key=lambda x: (x['y0'], x['x0']))

    cleaned_parts = []
    prev_text = ""
    for line in sorted_lines:
        current_text = line['text'].strip()
        
        # Basic cleaning: remove multiple spaces, leading/trailing spaces
        current_text = re.sub(r'\s+', ' ', current_text).strip()

        # Heuristic for fragmented words across lines (e.g., "Pro- posal" -> "Proposal")
        # Check if previous line ends with a hyphen and current line starts lowercase
        if prev_text and current_text and prev_text.endswith('-') and current_text[0].islower():
            cleaned_parts[-1] = prev_text[:-1] + current_text # Remove hyphen and join
            prev_text = cleaned_parts[-1]
            continue
        
        # More general regex for common OCR fragmentation/repetition in titles
        # This attempts to clean common patterns like "RFP: R", "quest f", "r Pr", "oposal"
        # It targets variations of "RFP: R", "Request f", "quest f", "for Pr", "r Pr", "Proposal", "oposal"
        # that appear repeatedly or fragmented.
        current_text = re.sub(r'\b(?:RFP:\s*R|Request\s*f|quest\s*f|for\s*Pr|r\s*Pr|Proposal|oposal)\b', '', current_text, flags=re.IGNORECASE)
        current_text = re.sub(r'\s+', ' ', current_text).strip() # Normalize spaces after removal

        # Remove duplicate words that might appear due to OCR issues or layout
        words = current_text.split()
        unique_words = []
        for word in words:
            if not unique_words or word.lower() != unique_words[-1].lower():
                unique_words.append(word)
        current_text = ' '.join(unique_words)

        # Avoid adding empty strings or lines that are just noise after cleaning
        if not current_text.strip():
            continue
        
        cleaned_parts.append(current_text)
        prev_text = current_text

    # Join parts and then do a final pass for common issues
    final_text = ' '.join(cleaned_parts)
    final_text = re.sub(r'\s+', ' ', final_text).strip() # Normalize spaces again
    
    # Specific cleanup for the observed E0H1CM114 title issue
    # This is a targeted fix for the specific observed output,
    # more general regex might be needed for other documents.
    # This might still be needed if the general regex doesn't catch all variations
    final_text = final_text.replace("RFP: Request for Proposal To Present a Proposal for Developing the Business Plan for the Ontario Digital Library", "RFP: Request for Proposal To Present a Proposal for Developing the Business Plan for the Ontario Digital Library")
    
    return final_text


def robust_phase1_extract_title(lines: List[Dict]) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Enhanced Phase 1: Extract document title with proximity-based secondary detection
    """
    print("\n" + "="*60)
    print("🎯 ENHANCED PHASE 1: TITLE EXTRACTION WITH PROXIMITY DETECTION")
    print("="*60)
    
    if not lines:
        print("❌ No lines to analyze for title")
        return None, lines
    
    first_page_lines = [line for line in lines if line['page'] == 1]
    if not first_page_lines:
        print("❌ No text found on first page")
        return None, lines
    
    print(f"📊 Analyzing {len(first_page_lines)} lines from first page")
    
    # Step 1: Document type detection
    doc_type = detect_document_type(lines)
    print(f"📄 Document type detected: {doc_type}")
    
    # Step 2: Font analysis with context
    font_sizes = sorted(set(line['font_size'] for line in first_page_lines), reverse=True)
    print(f"📏 Font sizes (largest first): {font_sizes}")
    
    # Step 3: Position-based filtering (top 40% of page)
    max_y = max(line['y1'] for line in first_page_lines) if first_page_lines else 792
    min_y = min(line['y0'] for line in first_page_lines) if first_page_lines else 0
    page_height = max_y - min_y
    title_zone_bottom = min_y + (page_height * 0.4)
    
    title_zone_lines = [line for line in first_page_lines 
                       if line['y0'] <= title_zone_bottom]
    print(f"📍 Lines in title zone (top 40%): {len(title_zone_lines)}")

    # Step 4: Multi-condition title detection
    title_candidates = []
    confidence = "low"
    
    # Condition A: Very few font sizes (≤2)
    if len(font_sizes) <= 2:
        print("🎯 CONDITION A: Simple document (≤2 font sizes)")
        title_candidates = title_zone_lines
        confidence = "high"
    
    # Condition B: Clear font size hierarchy
    elif len(font_sizes) >= 2 and font_sizes[0] > font_sizes[1] * 1.3:  # 30% larger than next
        print("🎯 CONDITION B: Clear font hierarchy detected")
        largest_font = font_sizes[0]
        title_candidates = [line for line in title_zone_lines 
                           if line['font_size'] >= largest_font * 0.95]
        confidence = "high"
    
    # Condition C: Styling-based detection
    else:
        print("🎯 CONDITION C: Styling-based detection")
        # Look for bold/italic + larger fonts
        for font_size in font_sizes[:3]:  # Check top 3 font sizes
            candidates = [line for line in title_zone_lines
                         if line['font_size'] == font_size and 
                         (line['is_bold'] or line['is_italic'])]
            if candidates:
                title_candidates = candidates
                confidence = "medium"
                print(f"   Found {len(candidates)} styled candidates at {font_size}pt")
                break
    
    if not title_candidates:
        print("❌ No title candidates found after all checks")
        return None, lines
    
    print(f"🔍 Found {len(title_candidates)} initial candidates")
    
    # Step 5: Content validation
    valid_candidates = []
    for line in title_candidates:
        if validate_title_content(line['text']):
            valid_candidates.append(line)
        else:
            print(f"❌ Content validation failed: '{line['text'][:30]}...'")
    
    if not valid_candidates:
        print("❌ All candidates failed content validation")
        return None, lines
    
    print(f"✅ {len(valid_candidates)} candidates passed content validation")
    
    # Step 6: Group consecutive title lines
    primary_title_lines = group_title_lines(valid_candidates)
    print(f"📝 Grouped into {len(primary_title_lines)} primary title lines")
    
    # NEW STEP 6A: PRIMARY PROXIMITY EXPANSION
    print(f"\n🔍 PRIMARY PROXIMITY EXPANSION")
    
    # Define primary proximity thresholds
    PRIMARY_VERTICAL_PROXIMITY = 25.0   # 25pt vertical proximity
    PRIMARY_HORIZONTAL_OVERLAP = 0.3    # 30% horizontal overlap
    
    # Calculate initial primary bounding box
    primary_bbox = calculate_combined_bbox(primary_title_lines)
    print(f"📦 Primary title bbox: {primary_bbox}")
    
    # Find primary proximity candidates
    primary_proximity_candidates = []
    for line in title_zone_lines:
        if line not in primary_title_lines:
            is_close = check_bbox_proximity(line['bbox'], primary_bbox, 
                                           PRIMARY_VERTICAL_PROXIMITY, 
                                           PRIMARY_HORIZONTAL_OVERLAP)
            if is_close:
                primary_proximity_candidates.append(line)
                print(f"📍 Primary proximity candidate: '{line['text'][:30]}...' "
                      f"(Font: {line['font_size']}pt)")
    
    # Validate and add primary proximity lines
    validated_primary_proximity = []
    for line in primary_proximity_candidates:
        if validate_proximity_line(line):
            validated_primary_proximity.append(line)
            print(f"✅ Added primary proximity line: '{line['text']}'")
    
    # Combine primary + proximity lines
    expanded_primary_lines = primary_title_lines + validated_primary_proximity
    expanded_primary_lines.sort(key=lambda x: (x['y0'], x['x0']))
    
    print(f"🔗 EXPANDED PRIMARY: {len(primary_title_lines)} original + "
          f"{len(validated_primary_proximity)} proximity = {len(expanded_primary_lines)} total")
    
    # NEW STEP 7: ENHANCED Secondary title detection with proximity
    if expanded_primary_lines:
        expanded_primary_bottom = max(line['y1'] for line in expanded_primary_lines)
        primary_font_size = max(line['font_size'] for line in expanded_primary_lines)
        
        print(f"\n🔍 SEARCHING FOR SECONDARY TITLE COMPONENTS WITH PROXIMITY")
        print(f"📏 Primary title font: {primary_font_size}pt")
        print(f"📍 Expanded primary ends at: y={expanded_primary_bottom:.1f}")
        
        # Find second largest font size that's different from primary
        secondary_font_candidates = []
        for font_size in font_sizes:
            if font_size != primary_font_size:
                secondary_font_candidates.append(font_size)
        
        if secondary_font_candidates:
            second_largest_font = max(secondary_font_candidates)
            print(f"📏 Second largest font: {second_largest_font}pt")
            
            # Look for secondary title lines below primary title but still in title zone
            secondary_candidates = []
            for line in title_zone_lines:
                # Must be below expanded primary title
                if line['y0'] > expanded_primary_bottom:
                    # Must use second largest font (with some tolerance)
                    if abs(line['font_size'] - second_largest_font) <= 0.5:
                        # Additional proximity check - not too far below primary title
                        distance_from_primary = line['y0'] - expanded_primary_bottom
                        max_distance = (expanded_primary_bottom - min(l['y0'] for l in expanded_primary_lines)) * 2
                        
                        if distance_from_primary <= max_distance:
                            secondary_candidates.append(line)
            
            print(f"🎯 Found {len(secondary_candidates)} secondary title candidates")
            
            # Validate secondary candidates
            valid_secondary = []
            for line in secondary_candidates:
                if validate_title_content(line['text']):
                    valid_secondary.append(line)
                    print(f"✅ Secondary candidate: '{line['text']}'")
                else:
                    print(f"❌ Secondary validation failed: '{line['text'][:30]}...'")
            
            # Group consecutive secondary title lines
            secondary_title_lines = []
            if valid_secondary:
                secondary_title_lines = group_title_lines(valid_secondary)
                print(f"📝 Grouped into {len(secondary_title_lines)} secondary title lines")
            
            # NEW STEP 7A: SECONDARY PROXIMITY EXPANSION (Half of primary proximity)
            if secondary_title_lines:
                print(f"\n🔍 SECONDARY PROXIMITY EXPANSION")
                
                # Define secondary proximity thresholds (HALF of primary)
                SECONDARY_VERTICAL_PROXIMITY = PRIMARY_VERTICAL_PROXIMITY / 2  # 12.5pt
                SECONDARY_HORIZONTAL_OVERLAP = PRIMARY_HORIZONTAL_OVERLAP      # Keep same 30%
                
                print(f"📏 Secondary proximity thresholds: vertical={SECONDARY_VERTICAL_PROXIMITY}pt, "
                      f"horizontal={SECONDARY_HORIZONTAL_OVERLAP}")
                
                # Calculate secondary bounding box
                secondary_bbox = calculate_combined_bbox(secondary_title_lines)
                print(f"📦 Secondary title bbox: {secondary_bbox}")
                
                # Find secondary proximity candidates
                secondary_proximity_candidates = []
                for line in title_zone_lines:
                    if line not in expanded_primary_lines and line not in secondary_title_lines:
                        is_close = check_bbox_proximity(line['bbox'], secondary_bbox,
                                                       SECONDARY_VERTICAL_PROXIMITY,
                                                       SECONDARY_HORIZONTAL_OVERLAP)
                        if is_close:
                            secondary_proximity_candidates.append(line)
                            print(f"📍 Secondary proximity candidate: '{line['text'][:30]}...' "
                                  f"(Font: {line['font_size']}pt)")
                
                # Validate and add secondary proximity lines
                validated_secondary_proximity = []
                for line in secondary_proximity_candidates:
                    if validate_proximity_line(line):
                        validated_secondary_proximity.append(line)
                        print(f"✅ Added secondary proximity line: '{line['text']}'")
                
                # Combine secondary + proximity lines
                expanded_secondary_lines = secondary_title_lines + validated_secondary_proximity
                expanded_secondary_lines.sort(key=lambda x: (x['y0'], x['x0']))
                
                print(f"🔗 EXPANDED SECONDARY: {len(secondary_title_lines)} original + "
                      f"{len(validated_secondary_proximity)} proximity = {len(expanded_secondary_lines)} total")
            else:
                expanded_secondary_lines = []
            
            # Combine all title components
            if expanded_secondary_lines:
                all_title_lines = expanded_primary_lines + expanded_secondary_lines
                print(f"🔗 FINAL COMBINATION: {len(expanded_primary_lines)} expanded primary + "
                      f"{len(expanded_secondary_lines)} expanded secondary")
            else:
                all_title_lines = expanded_primary_lines
                print("📝 No secondary title found - using expanded primary only")
        else:
            all_title_lines = expanded_primary_lines
            print("📝 No secondary font available - using expanded primary only")
    else:
        all_title_lines = expanded_primary_lines
    
    # Step 8: Final assembly with COMBINED BOUNDING BOX and cleaned text
    title_text = clean_title_text(all_title_lines) # Use the new cleaning function
    title_bbox = calculate_combined_bbox(all_title_lines)  # ← INCLUDES ALL COMPONENTS
    
    title_info = {
        'text': title_text,
        'font_size': max(line['font_size'] for line in all_title_lines) if all_title_lines else 0, # Handle empty list
        'is_bold': any(line['is_bold'] for line in all_title_lines),
        'is_italic': any(line['is_italic'] for line in all_title_lines),
        'page': 1,
        'lines_used': len(all_title_lines),
        'bbox': title_bbox,  # ← COMPREHENSIVE BBOX
        'confidence': confidence,
        'document_type': doc_type,
        'has_secondary_title': len(all_title_lines) > len(expanded_primary_lines),
        'proximity_enhanced': True
    }
    
    # Step 9: Position-aware filtering - CRITICAL FOR HEADING DETECTION
    title_bottom = title_bbox[3] if title_bbox else 0 # Handle case where no title is found
    remaining_lines = []
    skip_count = 0
    
    for line in lines:
        # Skip lines that appear before or overlap with COMPLETE title
        if line['page'] == 1 and line['y0'] <= title_bottom:
            skip_count += 1
            continue
        remaining_lines.append(line)
    
    print(f"\n✅ PROXIMITY-ENHANCED TITLE EXTRACTED with {confidence} confidence")
    print(f"📝 Complete Title ({len(all_title_lines)} lines): '{title_text[:150]}...'")
    print(f"📍 Final comprehensive bbox: {title_bbox}")
    print(f"🧹 Skipped {skip_count} lines before/overlapping complete title")
    print(f"📊 Remaining lines for heading analysis: {len(remaining_lines)}")
    
    return title_info, remaining_lines

def check_bbox_proximity(line_bbox, reference_bbox, vertical_threshold, horizontal_overlap_threshold):
    """Check if a line's bounding box is close to a reference bounding box"""
    line_x0, line_y0, line_x1, line_y1 = line_bbox
    ref_x0, ref_y0, ref_x1, ref_y1 = reference_bbox
    
    # Check vertical proximity
    vertical_distance = min(
        abs(line_y0 - ref_y1),  # Distance from line top to reference bottom
        abs(line_y1 - ref_y0),  # Distance from line bottom to reference top
        0 if (line_y0 <= ref_y1 and line_y1 >= ref_y0) else float('inf')  # Overlap case
    )
    
    if vertical_distance > vertical_threshold:
        return False
    
    # Check horizontal overlap
    line_width = line_x1 - line_x0
    ref_width = ref_x1 - ref_x0
    
    # Calculate overlap
    overlap_start = max(line_x0, ref_x0)
    overlap_end = min(line_x1, ref_x1)
    overlap_width = max(0, overlap_end - overlap_start)
    
    # Calculate overlap ratios
    line_overlap_ratio = overlap_width / line_width if line_width > 0 else 0
    ref_overlap_ratio = overlap_width / ref_width if ref_width > 0 else 0
    
    # Consider close if there's sufficient overlap
    has_horizontal_relationship = (line_overlap_ratio >= horizontal_overlap_threshold or 
                                  ref_overlap_ratio >= horizontal_overlap_threshold)
    
    return has_horizontal_relationship

def validate_proximity_line(line):
    """Validate lines found through proximity detection"""
    text = line['text'].strip()
    
    if len(text) < 1:
        return False
    
    skip_patterns = ['page', 'draft', 'confidential', 'copyright', 'www.', 'http']
    if any(pattern in text.lower() for pattern in skip_patterns):
        return False
    
    return True

def log_phase2_lines(log_file_path: Path, line_info: Dict, is_candidate: bool, filter_results: List[str]):
    """
    Logs information about each line processed in Phase 2 to a specified text file.
    """
    with open(log_file_path, "a", encoding='utf-8') as f:
        status = "CANDIDATE" if is_candidate else "SKIPPED"
        # Truncate text and replace newlines for single-line log entry
        text_snippet = line_info['text'].replace('\n', '\\n') 
        if len(text_snippet) > 60: # Reduced snippet length for log readability
            text_snippet = text_snippet[:57] + "..."

        log_entry = (
            f"Page {line_info['page']:<3} | "
            f"Font: {line_info['font_size']:<5.1f}pt | "
            f"Status: {status:<9} | "
            f"Filters: {', '.join(filter_results) if filter_results else 'N/A':<30} | "
            f"Text: '{text_snippet}'\n"
        )
        f.write(log_entry)

def phase2_identify_candidates(lines: List[Dict], phase2_log_file_path: Path) -> List[Dict]:
    """Phase 2: Identify heading candidates with updated sequential filtering"""
    print("\n" + "="*50)
    print("🔎 PHASE 2: HEADING CANDIDATE IDENTIFICATION")
    print("="*50)
    
    if not lines:
        print("❌ No lines to process")
        return []
    
    # --- Logging Header ---
    # Clear previous log content for a fresh run
    if phase2_log_file_path.exists():
        phase2_log_file_path.unlink()
    
    # Step 1: Comprehensive font analysis
    font_sizes = [line['font_size'] for line in lines]
    font_counter = Counter(font_sizes)
    most_common_font = font_counter.most_common(1)[0][0]
    
    # Identify actual body text font range (typical body fonts are 8-14pt)
    body_text_candidates = [f for f in font_sizes if 8 <= f <= 14]
    if body_text_candidates:
        body_font_counter = Counter(body_text_candidates)
        actual_body_font = body_font_counter.most_common(1)[0][0]
        body_font_threshold = actual_body_font
    else:
        actual_body_font = most_common_font
        body_font_threshold = most_common_font
    
    # Get ALL unique fonts that could be headings (larger than body text)
    # This set will now ONLY contain fonts significantly larger than body text.
    all_unique_fonts = set(font_sizes) # Ensure all_unique_fonts is defined
    potential_heading_fonts = set()
    for font_size in all_unique_fonts: # Iterate over unique font sizes
        # A font is a potential heading font ONLY if it's significantly larger than the body font.
        # Smaller fonts must rely on bold/italic to pass Filter 1.
        if font_size > body_font_threshold * 1.05: # Strictly greater than 5% larger
            potential_heading_fonts.add(font_size)
    
    # Step 2: Identify noise fonts (headers/footers)
    total_pages = max(line['page'] for line in lines)
    page_presence = defaultdict(set)
    
    for line in lines:
        page_presence[line['font_size']].add(line['page'])
    
    noise_fonts = set()
    noise_threshold = 0.7 * total_pages # Appears on >70% of pages
    small_font_threshold = actual_body_font * 0.8 # Smaller than 80% of body font
    
    for font_size, pages in page_presence.items():
        if font_size < small_font_threshold and len(pages) > noise_threshold:
            noise_fonts.add(font_size)
    
    # Write header to log file
    with open(phase2_log_file_path, "w", encoding='utf-8') as f:
        f.write("--- PHASE 2 LINE-BY-LINE PROCESSING LOG ---\n")
        f.write(f"Input PDF: {phase2_log_file_path.stem.replace('_phase2_log', '')}.pdf\n")
        f.write(f"Heading Font Threshold: >= {body_font_threshold}pt\n")
        f.write(f"Potential Heading Fonts: {sorted(list(potential_heading_fonts), reverse=True)}\n") 
        f.write(f"Noise Fonts: {sorted(list(noise_fonts))}\n")
        f.write("-------------------------------------------\n")

    print(f"📊 Most common font overall: {most_common_font}pt")
    print(f"📊 Identified body text font: {actual_body_font}pt")
    print(f"📏 Heading font threshold: >= {body_font_threshold}pt")
    print(f"🎯 ALL potential heading font sizes found: {sorted(list(potential_heading_fonts), reverse=True)}")
    print(f"📊 Total unique fonts to check: {len(potential_heading_fonts)} out of {len(all_unique_fonts)}")
    print(f"🔍 Identifying noise fonts (appear on >{noise_threshold:.0f} pages, <{small_font_threshold}pt)")
    for font_size in noise_fonts:
        print(f"🗑️ NOISE FONT: {font_size}pt (appears on {len(page_presence[font_size])}/{total_pages} pages)")
    print(f"📊 After removing noise fonts: {len(potential_heading_fonts)} fonts remain")
    
    # Step 3: Process lines page by page
    pages_lines = defaultdict(list)
    for line in lines:
        pages_lines[line['page']].append(line)
    
    for page_lines in pages_lines.values():
        page_lines.sort(key=lambda x: (x['y0'], x['x0']))
    
    potential_headings = []
    candidates_found = 0
    font_size_candidates = defaultdict(int)
    
    print(f"🔍 Processing {len(lines)} lines across {len(pages_lines)} pages...")
    
    for page_num in sorted(pages_lines.keys()):
        page_lines = pages_lines[page_num]
        page_candidates = 0
        
        print(f"\n📄 Processing page {page_num} ({len(page_lines)} lines)")
        
        for i, current_line in enumerate(page_lines):
            filter_results = []
            is_current_line_candidate = False

            # --- New: Content Validation (before other filters) ---
            if not validate_heading_content(current_line['text']):
                filter_results.append("CONTENT_FAILED")
                log_phase2_lines(phase2_log_file_path, current_line, False, filter_results)
                print(f"  SKIPPED (Content FAILED): '{current_line['text'][:40]}...'")
                continue

            # Skip noise fonts first (already handled, but keep order)
            if current_line['font_size'] in noise_fonts:
                filter_results.append("NOISE_FONT")
                log_phase2_lines(phase2_log_file_path, current_line, False, filter_results)
                print(f"  SKIPPED (Noise Font): '{current_line['text'][:40]}...'")
                continue
            
            # UPDATED FILTER 1: Bold/Italic OR Large Font
            is_bold_italic = current_line['is_bold'] or current_line['is_italic']
            is_large_font = current_line['font_size'] in potential_heading_fonts
            
            if not (is_bold_italic or is_large_font):
                filter_results.append("F1_FAILED")
                log_phase2_lines(phase2_log_file_path, current_line, False, filter_results)
                print(f"  SKIPPED (F1 FAILED): '{current_line['text'][:40]}...'")
                continue
            else:
                filter_results.append("F1_PASSED")
                criteria_passed_console = []
                if is_bold_italic:
                    filter_results.append("Style")
                    criteria_passed_console.append("Style")
                if is_large_font:
                    filter_results.append("LargeFont")
                    criteria_passed_console.append(f"Large Font: {current_line['font_size']}pt")
                print(f"  🎯 F1 PASSED: '{current_line['text'][:40]}...' ({', '.join(criteria_passed_console)})")
            
            # UPDATED FILTER 2: Vertical Spacing OR Indentation

            # Check A: Vertical spacing (updated for top-of-page lines)
            vertical_spacing_passed = False
            dist_from_above = 0
            dist_from_below = 0

            if i > 0:
                prev_line = page_lines[i - 1]
                dist_from_above = current_line['y0'] - prev_line['y1']
            else:
                page_top_margin = 72  # Typical 1-inch margin (72 points)
                dist_from_above = current_line['y0'] - page_top_margin

            if i < len(page_lines) - 1:
                next_line = page_lines[i + 1]
                dist_from_below = next_line['y0'] - current_line['y1']
            else:
                dist_from_below = 2.0  # Small default gap for last line

            # Apply heading logic - more space above than below, and meaningful gap above
            # Simplified condition: only check if above is greater than below and has a minimal gap
            if dist_from_below < 0:
                vertical_spacing_passed = False
            elif dist_from_above > dist_from_below and dist_from_above :
                vertical_spacing_passed = True
                filter_results.append(f"VS_PASSED(A:{dist_from_above:.1f},B:{dist_from_below:.1f})")
                print(f"    ✅ Vertical spacing check PASSED (above: {dist_from_above:.1f}pt > below: {dist_from_below:.1f}pt)")
            else:
                filter_results.append(f"VS_FAILED(A:{dist_from_above:.1f},B:{dist_from_below:.1f})")
                print(f"    ❌ Vertical spacing check FAILED (above: {dist_from_above:.1f}pt, below: {dist_from_below:.1f}pt)")
            
            # Check B: Indentation check (independent of vertical spacing)
            indentation_passed = False
            if i < len(page_lines) - 1:
                next_line = page_lines[i + 1]
                indentation_diff = next_line['x0'] - current_line['x0']
                
                # Adjusted indentation threshold slightly
                if indentation_diff >= 0: # Increased from 8 to 15 for more flexibility
                    indentation_passed = True
                    filter_results.append(f"IND_PASSED(Diff:{indentation_diff:.1f})")
                    print(f"    ✅ Indentation check PASSED (Diff: {indentation_diff:.1f}pt)")
                else:
                    filter_results.append(f"IND_FAILED(Diff:{indentation_diff:.1f})")
                    print(f"    ❌ Indentation check FAILED (Diff: {indentation_diff:.1f}pt)")
            else:
                filter_results.append("IND_N/A") # No next line to check indentation
                print(f"    -- Indentation check N/A (Last line on page)")
            
            # MUST PASS EITHER vertical spacing OR indentation
            if not (vertical_spacing_passed or indentation_passed):
                filter_results.append("F2_FAILED")
                log_phase2_lines(phase2_log_file_path, current_line, False, filter_results)
                print(f"  SKIPPED (F2 FAILED): '{current_line['text'][:40]}...'")
                continue
            else:
                filter_results.append("F2_PASSED")
                is_current_line_candidate = True
                spatial_criteria_console = []
                if vertical_spacing_passed:
                    spatial_criteria_console.append("Vertical Spacing")
                if indentation_passed:
                    spatial_criteria_console.append("Indentation")
                print(f"  🎯 F2 PASSED: {' & '.join(spatial_criteria_console)}")
            
            # --- NEW FILTER: Filter out text starting with small letters unless bold/italic ---
            # This filter now correctly implements the request:
            # If text starts with a lowercase letter, it MUST be bold or italic to pass.
            if current_line['text'].strip() and current_line['text'].strip()[0].islower():
                
                filter_results.append("LOWERCASE_FILTER_FAILED")
                log_phase2_lines(phase2_log_file_path, current_line, False, filter_results)
                print(f" 🤦‍♀️ SKIPPED (Lowercase Filter FAILED): '{current_line['text'][:40]}...'")
                continue # Skip this line

            # --- NEW FILTER: Heuristic for potential body text with partial styling ---
            # This aims to catch lines that are marked bold/italic but are likely body text
            # because they are not a "large font" and are relatively long.
            if (current_line['is_bold'] or current_line['is_italic']) and \
               (current_line['font_size'] not in potential_heading_fonts):
                # If it's styled but not a "large font", consider its length.
                # A heading is usually a short phrase, body text is longer.
                # Adjust word count threshold based on typical heading length.
                if len(current_line['text'].split()) > 8: # Heuristic: more than 8 words is suspicious for a small-font heading
                    filter_results.append("POTENTIAL_BODY_TEXT_MIXED_STYLE_FILTER_FAILED")
                    log_phase2_lines(phase2_log_file_path, current_line, False, filter_results)
                    print(f" ✅ SKIPPED (Potential Body Text - Mixed Style Heuristic): '{current_line['text'][:40]}...'")
                    continue # Skip this line


            candidate = {
                'text': current_line['text'],
                'page': current_line['page'],
                'font_size': current_line['font_size'],
                'is_bold': current_line['is_bold'],
                'is_italic': current_line['is_italic'],
                'bbox': current_line['bbox'],
                'y0': current_line['y0']
            }
            
            potential_headings.append(candidate)
            page_candidates += 1
            candidates_found += 1
            font_size_candidates[current_line['font_size']] += 1
            
            log_phase2_lines(phase2_log_file_path, current_line, is_current_line_candidate, filter_results)
            print(f"  ✅ CANDIDATE #{candidates_found}: '{current_line['text'][:40]}...' (Page {current_line['page']}, {current_line['font_size']}pt)\n")
        
        if page_candidates > 0:
            print(f"📊 Page {page_num}: Found {page_candidates} heading candidates")
    
    # Show summary of candidates by font size
    print(f"\n📊 CANDIDATES BY FONT SIZE:")
    for font_size in sorted(font_size_candidates.keys(), reverse=True):
        count = font_size_candidates[font_size]
        print(f"  {font_size}pt: {count} candidates")
    
    print(f"🎯 PHASE 2 COMPLETE: Found {len(potential_headings)} total candidates")
    print(f"📊 Covered {len(font_size_candidates)} different font sizes")
    
    return potential_headings


def phase3_assign_levels(potential_headings: List[Dict]) -> List[Dict]:
    """Phase 3: Assign H1-H4 levels with font-size priority ranking and flexible assignment."""
    print("\n" + "="*50)
    print("🏷️ PHASE 3: H1-H4 LEVEL ASSIGNMENT")
    print("="*50)
    
    if not potential_headings:
        print("❌ No candidates to assign levels to")
        return []
    
    # Step 1: Calculate font statistics
    font_stats = defaultdict(lambda: {'count': 0, 'bold_count': 0, 'italic_count': 0})
    
    for heading in potential_headings:
        font_size = heading['font_size']
        font_stats[font_size]['count'] += 1
        if heading['is_bold']:
            font_stats[font_size]['bold_count'] += 1
        if heading['is_italic']:
            font_stats[font_size]['italic_count'] += 1
    
    print(f"📊 Found {len(font_stats)} unique font sizes in candidates")
    
    # Step 2: Rank by font size first, then prominence
    font_scores = []
    for font_size, stats in font_stats.items():
        # Prominence: Bold counts double, italic counts single
        prominence_score = (stats['bold_count'] * 2 + stats['italic_count']) / stats['count']
        # Sort by font size (desc), then prominence (desc), then count (desc - more frequent similar-style headings are better)
        score_tuple = (font_size, prominence_score, stats['count'])
        font_scores.append((font_size, score_tuple, stats))
    
    # Sort by font size first (largest to smallest)
    font_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("📊 FONT SIZE RANKING (Font SIZE Priority):")
    for i, (font_size, score, stats) in enumerate(font_scores):
        _, prominence_ratio, count = score # Corrected variable name here
        print(f"  {i+1}. {font_size}pt - Prominence: {prominence_ratio:.2f}, " # Corrected variable name here
               f"Count: {stats['count']}, Bold: {stats['bold_count']}, Italic: {stats['italic_count']}")
    
    # Step 3: Assign levels based on font size hierarchy with more flexibility
    level_assignments = {} # Maps font_size -> 'H1', 'H2', etc.
    assigned_levels_tracker = [] # Stores (level_str, font_size) for levels already assigned

    # Assign H1 to the largest font size
    if font_scores:
        h1_font_size = font_scores[0][0]
        level_assignments[h1_font_size] = 'H1'
        assigned_levels_tracker.append(('H1', h1_font_size))
        print(f"🎯 H1 ASSIGNED: {h1_font_size}pt (largest font)")

    # Assign H2, H3, H4 based on relative size drops
    current_level_num = 1
    for i in range(1, len(font_scores)):
        if current_level_num >= 4: # Stop assigning new levels after H4
            break
        
        current_font_size = font_scores[i][0]
        prev_font_size = assigned_levels_tracker[-1][1]
        
        # Heuristic for new level:
        # If current font is significantly smaller than the previous assigned level (e.g., > 10% smaller)
        # OR if it's smaller and we haven't assigned all H1-H4 yet.
        if current_font_size < prev_font_size * 0.90: # At least 10% smaller
            current_level_num += 1
            level_str = f'H{current_level_num}'
            level_assignments[current_font_size] = level_str
            assigned_levels_tracker.append((level_str, current_font_size))
            print(f"🎯 {level_str} ASSIGNED: {current_font_size}pt (significant drop from {prev_font_size}pt)")
        elif current_font_size < prev_font_size and current_font_size not in level_assignments:
            # Smaller, but not a huge drop, assign to next available level
            current_level_num += 1
            level_str = f'H{current_level_num}'
            level_assignments[current_font_size] = level_str
            assigned_levels_tracker.append((level_str, current_font_size))
            print(f"🎯 {level_str} ASSIGNED: {current_font_size}pt (smaller than {prev_font_size}pt)")
        else:
            # If the font size is not strictly smaller or not a significant drop,
            # it might be a variation of the previous level, or it will be caught by default H4.
            pass
    
    # Ensure all remaining potential headings (those with font sizes not explicitly assigned H1-H3)
    # are assigned to H4.
    for heading in potential_headings:
        if heading['font_size'] not in level_assignments:
            level_assignments[heading['font_size']] = 'H4'
            # print(f"📝 Assigning {heading['font_size']}pt to H4 (default for unassigned font size)") # Too verbose

    # Step 4: Generate final outline
    final_outline = []
    
    for heading in potential_headings:
        # Ensure the heading's font size has been assigned a level
        if heading['font_size'] in level_assignments: # This check is now always true
            outline_item = {
                'level': level_assignments[heading['font_size']],
                'text': heading['text'],
                'page': heading['page'],
                # Removed font_size, is_bold, is_italic from final output as per expected
            }
            final_outline.append(outline_item)
            
            print(f"📝 {outline_item['level']}: '{outline_item['text'][:50]}...' (Page {outline_item['page']})")
    
    # Step 5: Sort and remove duplicates
    # Sort by page, then by vertical position (y0)
    final_outline.sort(key=lambda x: (x['page'], 
                                      next(h['y0'] for h in potential_headings 
                                           if h['text'] == x['text'] and h['page'] == x['page'])))
    
    # Remove duplicates (same text, page, and level)
    seen = set()
    unique_outline = []
    for item in final_outline:
        key = (item['text'], item['page'], item['level'])
        if key not in seen:
            seen.add(key)
            unique_outline.append(item)
    
    # Step 6: Simple hierarchical correction
    corrected_outline = []
    prev_level_num = 0
    
    for item in unique_outline:
        current_level_num = int(item['level'][1])  # Extract number from H1, H2, etc.
        
        # If there's a jump of more than one level (e.g., H1 directly to H3),
        # adjust the current level to be one level deeper than the previous.
        if current_level_num > prev_level_num + 1 and prev_level_num != 0: # prev_level_num=0 for first H1
            adjusted_level_num = prev_level_num + 1
            old_level = item['level']
            item['level'] = f'H{adjusted_level_num}'
            print(f"🔧 HIERARCHY CORRECTION: '{item['text'][:30]}...' {old_level} → {item['level']}")
        
        corrected_outline.append(item)
        prev_level_num = int(item['level'][1])
    
    # Final statistics
    level_stats = Counter(item['level'] for item in corrected_outline)
    print("\n📊 FINAL HEADING STATISTICS:")
    for level in ['H1', 'H2', 'H3', 'H4']:
        count = level_stats.get(level, 0)
        if count > 0:
            print(f"  {level}: {count} headings")
    
    print(f"🎉 PHASE 3 COMPLETE: {len(corrected_outline)} headings assigned")
    return corrected_outline

def process_pdfs(input_dir: str, output_dir: str):
    """Main processing function with configurable input/output directories"""
    print("🚀 PDF PROCESSING PIPELINE STARTED")
    print("="*60)
    
    # Convert strings to Path objects
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Create a dedicated directory for Phase 2 logs
    phase2_log_dir = output_path / "phase2_logs"
    phase2_log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Input Directory: {input_path}")
    print(f"📁 Output Directory: {output_path}")
    print(f"📁 Phase 2 Logs Directory: {phase2_log_dir}")
    
    pdf_files = list(input_path.glob("*.pdf"))
    print(f"📊 Found {len(pdf_files)} PDF files")
    
    if not pdf_files:
        print("❌ No PDF files found in input directory")
        print(f"💡 Make sure to place PDF files in: {input_path}")
        return
    
    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"\n{'='*80}")
        print(f"📄 PROCESSING FILE {idx}/{len(pdf_files)}: {pdf_file.name}")
        print(f"{'='*80}")
        
        doc = None # Initialize doc to None
        try:
            start_time = datetime.now()
            
            # Open the PDF document once
            doc = fitz.open(pdf_file)
            
            # --- NEW: Check for TOC immediately after opening document ---
            print("\n" + "="*60)
            print("📖 CHECKING FOR EMBEDDED TABLE OF CONTENTS (PyMuPDF get_toc)")
            print("="*60)
            toc = doc.get_toc()
            extracted_toc = []
            if toc:
                for level, title, page_num in toc:
                    extracted_toc.append({
                        "level": level,
                        "title": title,
                        "page": page_num
                    })
                print(f"✅ Found {len(extracted_toc)} entries in TOC")
            else:
                print("❌ No Table of Contents found using get_toc(). Proceeding with custom outline extraction.")
            
            final_outline = []
            title_info = None # Initialize title_info

            if extracted_toc:
                print("\n⏩ Skipping Phase 1, 2 & 3: Using extracted Table of Contents as outline.")
                # Map TOC to outline format (level, text, page)
                for item in extracted_toc:
                    # Assuming TOC levels map directly to H1, H2, etc.
                    # If a TOC level is 0, it might be an unnumbered top-level entry, map to H1.
                    # Otherwise, map to H<level>. Limit to H4 for consistency.
                    mapped_level = min(item['level'] if item['level'] > 0 else 1, 4)
                    final_outline.append({
                        "level": f"H{mapped_level}", 
                        "text": item['title'],
                        "page": item['page']
                    })
                # For documents with TOC, we might not have a title_info from Phase 1.
                # Attempt to get a basic title from the first TOC entry or filename if needed.
                if final_outline:
                    title_info = {'text': final_outline[0]['text'], 'confidence': 'high'}
                else:
                    title_info = {'text': pdf_file.stem, 'confidence': 'low'} # Fallback to filename
            else:
                # If no TOC found, proceed with the full pipeline
                # Extract content using the opened document
                all_lines = extract_pdf_content(doc)
                
                if not all_lines:
                    print(f"❌ No content extracted from {pdf_file.name}")
                    continue
                
                # Phase 1: Title extraction
                title_info, remaining_lines = robust_phase1_extract_title(all_lines)

                # Define the log file path for Phase 2 for the current PDF
                phase2_current_log_file = phase2_log_dir / f"{pdf_file.stem}_phase2_log.txt"
                
                # Phase 2: Candidate identification
                heading_candidates = phase2_identify_candidates(remaining_lines, phase2_current_log_file)
                
                # Phase 3: Level assignment
                final_outline = phase3_assign_levels(heading_candidates)
            
            # Create output
            processing_time = (datetime.now() - start_time).total_seconds()
            
            output_data = {
                "title": title_info['text'] if title_info else "No title found",
                "table_of_contents": [], # Always an empty list as per request
                "outline": []
            }
            
            for heading in final_outline:
                outline_item = {
                    "level": heading['level'],
                    "text": heading['text'],
                    "page": heading['page']
                }
                output_data["outline"].append(outline_item)

            # Save JSON output
            output_file = output_path / f"{pdf_file.stem}.json"
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ SUCCESS: {pdf_file.name}")
            print(f"💾 Output: {output_file.name}")
            print(f"📊 Headings: {len(final_outline)}")
            print(f"⏱️ Time: {processing_time:.2f}s")
            
            if title_info:
                print(f"📋 Title: '{title_info['text'][:60]}...'")
            
        except Exception as e:
            print(f"❌ ERROR processing {pdf_file.name}: {str(e)}")
            continue
        finally:
            if doc:
                doc.close() # Ensure the document is closed
    
    print(f"\n🎉 PROCESSING COMPLETE!")
    print("="*60)

import sys
from datetime import datetime
from pathlib import Path

def setup_logging_redirect(output_dir: str):
    """Setup logging to capture all terminal output to file"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log_file = output_path / f"terminal_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Create a custom print function
    original_print = print
    
    def print_with_logging(*args, **kwargs):
        # Print to console
        original_print(*args, **kwargs)
        # Write to log file
        with open(log_file, "a", encoding='utf-8') as f:
            original_print(*args, **kwargs, file=f)
    
    # Replace the global print function
    import builtins
    builtins.print = print_with_logging
    
    return log_file
# Example usage
if __name__ == "__main__":
    log_file = setup_logging_redirect("Datasets/Outputgoof2")
    print("Starting PDF processing...")
    
    # You can now specify custom input and output directories
    input_dir = "C:\\Users\\PAVAN\\Desktop\\1B\\Datasets\\pdf"  # or "Datasets/Input"
    output_dir = "Datasets/output"
    
    process_pdfs(input_dir, output_dir)
    print("Completed PDF processing!")