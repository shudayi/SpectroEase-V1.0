import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from typing import Dict, Any, List, Optional
import os
import sys
from datetime import datetime

class SpectrumReportGenerator:
    """Spectrum analysis report generator"""
    
    def __init__(self, output_dir="reports"):
        """
        Initialize report generator
        
        Args:
            output_dir: Report output directory
        """
        # Handle exe environment - use absolute path
        if getattr(sys, 'frozen', False):
            # Running as exe
            exe_dir = os.path.dirname(sys.executable)
            self.output_dir = os.path.join(exe_dir, output_dir)
        else:
            # Running as script
            self.output_dir = os.path.abspath(output_dir)
        
        # Ensure directory exists
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        except PermissionError:
            # Fallback to user documents folder
            import tempfile
            user_docs = os.path.expanduser("~/Documents/SpectroEase")
            self.output_dir = os.path.join(user_docs, "reports")
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
    def generate_report(self, 
                       spectra_data: Dict[str, Any],
                       preprocessing_results: Dict[str, Any],
                       feature_extraction_results: Dict[str, Any],
                       analysis_results: Dict[str, Any],
                       title: str = "Spectral Analysis Report") -> str:
        """
        Generate analysis report
        
        Args:
            spectra_data: Spectral data information
            preprocessing_results: Preprocessing results
            feature_extraction_results: Feature extraction results
            analysis_results: Analysis results
            title: Report title
            
        Returns:
            str: Report file path
        """
        # Create PDF document
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{title}_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        
        # Prepare report content
        story = []
        styles = getSampleStyleSheet()
        
        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph(title, title_style))
        
        # Add timestamp
        timestamp_style = ParagraphStyle(
            'Timestamp',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.gray
        )
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                             timestamp_style))
        story.append(Spacer(1, 20))
        
        # 1. Data Overview
        story.append(Paragraph("1. Data Overview", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Add basic data information table
        data_info = [
            ["Data Dimensions", f"{spectra_data.get('shape', 'N/A')}"],
            ["Wavelength Range", f"{spectra_data.get('wavelength_range', 'N/A')}"],
            ["Number of Samples", f"{spectra_data.get('n_samples', 'N/A')}"],
            ["Number of Features", f"{spectra_data.get('n_features', 'N/A')}"]
        ]
        
        # Create table with header row
        t = Table(data_info, colWidths=[150, 350])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 20))
        
        # 2. Preprocessing Results
        story.append(Paragraph("2. Preprocessing Results", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Add preprocessing method information
        if preprocessing_results:
            preprocess_info = [["Method", "Parameters"]]  # Add header row
            for method, params in preprocessing_results.items():
                preprocess_info.append([method, str(params)])
                
            t = Table(preprocess_info, colWidths=[150, 350])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
        else:
            preprocess_info = [["No preprocessing methods applied"]]
            t = Table(preprocess_info, colWidths=[500])
            t.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Oblique'),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.grey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
        story.append(t)
        story.append(Spacer(1, 20))
        
        # 3. Feature Extraction Results
        story.append(Paragraph("3. Feature Extraction Results", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Add feature extraction method information
        if feature_extraction_results and 'No feature selection' not in feature_extraction_results:
            feature_info = [["Method", "Results"]]  # Add header row
            for method, params in feature_extraction_results.items():
                feature_info.append([method, str(params)])
                
            t = Table(feature_info, colWidths=[150, 350])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
        else:
            feature_info = [["No feature selection methods applied"]]
            t = Table(feature_info, colWidths=[500])
            t.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Oblique'),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.grey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
        story.append(t)
        story.append(Spacer(1, 20))
        
        # 4. Analysis Results
        story.append(Paragraph("4. Analysis Results", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Add analysis results information
        if analysis_results and 'No analysis' not in analysis_results:
            analysis_info = [["Metric", "Value"]]  # Add header row
            for metric, value in analysis_results.items():
                analysis_info.append([metric, str(value)])
                
            t = Table(analysis_info, colWidths=[150, 350])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
        else:
            analysis_info = [["No modeling or analysis has been performed yet"]]
            t = Table(analysis_info, colWidths=[500])
            t.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Oblique'),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.grey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
        story.append(t)
        
        # Generate PDF
        doc.build(story)
        return filepath 