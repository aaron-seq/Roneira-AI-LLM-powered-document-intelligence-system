# ==============================================================================
# Free OCR Service Implementation
# Using Tesseract OCR, PyMuPDF, and EasyOCR for document processing
# ==============================================================================

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import tempfile

# Free OCR libraries
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
import cv2
import numpy as np

# Optional: EasyOCR for better accuracy
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available, using Tesseract only")

logger = logging.getLogger(__name__)

class FreeOCRService:
    """Free OCR service using open source tools"""
    
    def __init__(self):
        self.tesseract_config = os.getenv("TESSERACT_CONFIG", "--psm 6")
        self.use_easyocr = os.getenv("USE_EASYOCR", "true").lower() == "true" and EASYOCR_AVAILABLE
        
        # Initialize EasyOCR if available
        if self.use_easyocr:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.use_easyocr = False
        
        logger.info(f"OCR Service initialized. Tesseract: ✓, EasyOCR: {'✓' if self.use_easyocr else '✗'}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            return thresh
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    async def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            pages_text = []
            images = []
            tables = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text
                text = page.get_text()
                pages_text.append({
                    "page": page_num + 1,
                    "text": text,
                    "word_count": len(text.split())
                })
                
                # Extract images for OCR
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("ppm")
                            
                            # Save to temp file for OCR
                            with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as temp_img:
                                temp_img.write(img_data)
                                temp_img_path = temp_img.name
                            
                            # Extract text from image
                            ocr_text = await self._extract_text_from_image(temp_img_path)
                            
                            images.append({
                                "page": page_num + 1,
                                "image_index": img_index,
                                "text": ocr_text,
                                "confidence": ocr_text.get("confidence", 0)
                            })
                            
                            # Clean up temp file
                            os.unlink(temp_img_path)
                        
                        pix = None
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
                
                # Extract tables (basic implementation)
                try:
                    tables_on_page = page.find_tables()
                    for table_index, table in enumerate(tables_on_page):
                        table_data = table.extract()
                        tables.append({
                            "page": page_num + 1,
                            "table_index": table_index,
                            "data": table_data,
                            "rows": len(table_data),
                            "columns": len(table_data[0]) if table_data else 0
                        })
                except Exception as e:
                    logger.debug(f"Table extraction failed for page {page_num + 1}: {e}")
            
            doc.close()
            
            return {
                "success": True,
                "pages": pages_text,
                "images": images,
                "tables": tables,
                "total_pages": len(pages_text),
                "total_text": " ".join([p["text"] for p in pages_text]),
                "extraction_method": "PyMuPDF"
            }
            
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "extraction_method": "PyMuPDF"
            }
    
    async def _extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract text from image using available OCR engines"""
        results = {
            "tesseract": None,
            "easyocr": None,
            "best_result": None,
            "confidence": 0
        }
        
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            processed_image = self._preprocess_image(image)
            
            # Tesseract OCR
            try:
                # Save processed image for tesseract
                temp_processed = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                cv2.imwrite(temp_processed.name, processed_image)
                
                # Extract text with confidence
                tesseract_data = pytesseract.image_to_data(
                    temp_processed.name, 
                    config=self.tesseract_config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Filter out low confidence text
                confident_text = []
                confidences = []
                
                for i, conf in enumerate(tesseract_data['conf']):
                    if int(conf) > 30:  # Confidence threshold
                        text = tesseract_data['text'][i].strip()
                        if text:
                            confident_text.append(text)
                            confidences.append(int(conf))
                
                tesseract_result = " ".join(confident_text)
                tesseract_confidence = np.mean(confidences) if confidences else 0
                
                results["tesseract"] = {
                    "text": tesseract_result,
                    "confidence": tesseract_confidence,
                    "word_count": len(confident_text)
                }
                
                os.unlink(temp_processed.name)
                
            except Exception as e:
                logger.warning(f"Tesseract OCR failed: {e}")
            
            # EasyOCR (if available)
            if self.use_easyocr:
                try:
                    easyocr_results = self.easyocr_reader.readtext(processed_image)
                    
                    easyocr_text = []
                    easyocr_confidences = []
                    
                    for (bbox, text, confidence) in easyocr_results:
                        if confidence > 0.3:  # Confidence threshold
                            easyocr_text.append(text)
                            easyocr_confidences.append(confidence)
                    
                    easyocr_result = " ".join(easyocr_text)
                    easyocr_confidence = np.mean(easyocr_confidences) if easyocr_confidences else 0
                    
                    results["easyocr"] = {
                        "text": easyocr_result,
                        "confidence": easyocr_confidence * 100,  # Convert to percentage
                        "word_count": len(easyocr_text)
                    }
                    
                except Exception as e:
                    logger.warning(f"EasyOCR failed: {e}")
            
            # Choose best result
            best_engine = "tesseract"
            best_confidence = results["tesseract"]["confidence"] if results["tesseract"] else 0
            
            if results["easyocr"] and results["easyocr"]["confidence"] > best_confidence:
                best_engine = "easyocr"
                best_confidence = results["easyocr"]["confidence"]
            
            if results[best_engine]:
                results["best_result"] = results[best_engine]
                results["best_engine"] = best_engine
                results["confidence"] = best_confidence
            
        except Exception as e:
            logger.error(f"Image OCR failed: {e}")
            results["error"] = str(e)
        
        return results
    
    async def extract_text_from_image_file(self, image_path: str) -> Dict[str, Any]:
        """Extract text from image file"""
        try:
            result = await self._extract_text_from_image(image_path)
            
            if result["best_result"]:
                return {
                    "success": True,
                    "text": result["best_result"]["text"],
                    "confidence": result["confidence"],
                    "engine_used": result.get("best_engine", "unknown"),
                    "word_count": result["best_result"]["word_count"],
                    "all_results": result
                }
            else:
                return {
                    "success": False,
                    "text": "",
                    "confidence": 0,
                    "error": "No text extracted",
                    "all_results": result
                }
                
        except Exception as e:
            logger.error(f"Image text extraction failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def extract_text_from_document(self, file_path: str) -> Dict[str, Any]:
        """Extract text from any supported document type"""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.pdf':
                return await self.extract_text_from_pdf(str(file_path))
            
            elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                return await self.extract_text_from_image_file(str(file_path))
            
            elif file_extension in ['.txt', '.md']:
                # Plain text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                return {
                    "success": True,
                    "text": text,
                    "word_count": len(text.split()),
                    "extraction_method": "direct_read"
                }
            
            elif file_extension == '.docx':
                # Word documents (requires python-docx)
                try:
                    from docx import Document
                    doc = Document(file_path)
                    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                    return {
                        "success": True,
                        "text": text,
                        "word_count": len(text.split()),
                        "extraction_method": "python-docx"
                    }
                except ImportError:
                    return {
                        "success": False,
                        "error": "python-docx not available for .docx files"
                    }
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {file_extension}"
                }
                
        except Exception as e:
            logger.error(f"Document text extraction failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of OCR services"""
        health = {
            "tesseract": False,
            "easyocr": False,
            "services_available": []
        }
        
        # Check Tesseract
        try:
            version = pytesseract.get_tesseract_version()
            health["tesseract"] = True
            health["tesseract_version"] = str(version)
            health["services_available"].append("tesseract")
        except Exception as e:
            logger.warning(f"Tesseract health check failed: {e}")
        
        # Check EasyOCR
        if self.use_easyocr:
            try:
                # Simple test
                test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
                _ = self.easyocr_reader.readtext(test_image)
                health["easyocr"] = True
                health["services_available"].append("easyocr")
            except Exception as e:
                logger.warning(f"EasyOCR health check failed: {e}")
        
        health["status"] = "healthy" if health["services_available"] else "unhealthy"
        return health

# Initialize global service instance
_ocr_service = None

def get_ocr_service() -> FreeOCRService:
    """Get or create OCR service instance"""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = FreeOCRService()
    return _ocr_service