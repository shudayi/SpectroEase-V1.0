# -*- coding: utf-8 -*-
"""
Algorithm Name Matcher Utility
Provides case-insensitive and normalized name matching for algorithms
"""

import re
from typing import Optional, Dict, Any


def normalize_algorithm_name(name: str) -> str:
    """
    Normalize algorithm name for matching:
    - Convert to lowercase
    - Remove extra spaces
    - Remove special characters that don't affect matching
    
    Args:
        name: Algorithm name to normalize
        
    Returns:
        Normalized name string
    """
    if not name:
        return ""
    
    # Remove emoji and special prefixes
    name = name.replace("🔧 ", "").replace("🔧", "")
    
    # Normalize whitespace
    name = re.sub(r'\s+', ' ', name.strip())
    
    # Convert to lowercase for case-insensitive matching
    normalized = name.lower()
    
    return normalized


def find_algorithm_by_name(name: str, plugins: Dict[str, Any], exact_match: bool = False) -> Optional[Any]:
    """
    Find algorithm in plugins dictionary by name with fuzzy matching
    
    Args:
        name: Algorithm name to search for
        plugins: Dictionary of algorithms (name -> algorithm)
        exact_match: If True, only do exact match. If False, try fuzzy matching
        
    Returns:
        Algorithm instance if found, None otherwise
    """
    if not name or not plugins:
        return None
    
    # Try exact match first
    if name in plugins:
        return plugins[name]
    
    # Try normalized exact match
    normalized_name = normalize_algorithm_name(name)
    for key, value in plugins.items():
        if normalize_algorithm_name(key) == normalized_name:
            return value
    
    if exact_match:
        return None
    
    # Try fuzzy matching (case-insensitive, space-insensitive)
    # Remove common words that might differ
    name_words = set(normalized_name.split())
    
    best_match = None
    best_score = 0
    
    for key, value in plugins.items():
        key_normalized = normalize_algorithm_name(key)
        key_words = set(key_normalized.split())
        
        # Calculate similarity score
        if name_words and key_words:
            # Jaccard similarity
            intersection = len(name_words & key_words)
            union = len(name_words | key_words)
            score = intersection / union if union > 0 else 0
            
            # Bonus for substring match
            if normalized_name in key_normalized or key_normalized in normalized_name:
                score += 0.3
            
            if score > best_score:
                best_score = score
                best_match = value
    
    # Only return if similarity is high enough (at least 50% match)
    if best_score >= 0.5:
        return best_match
    
    return None


def get_algorithm_name_variants(name: str) -> list:
    """
    Generate common variants of an algorithm name for matching
    
    Args:
        name: Algorithm name
        
    Returns:
        List of name variants
    """
    variants = [name]
    
    # Original with different cases
    variants.append(name.lower())
    variants.append(name.upper())
    variants.append(name.title())
    
    # Without spaces
    variants.append(name.replace(' ', ''))
    variants.append(name.replace(' ', '_'))
    variants.append(name.replace(' ', '-'))
    
    # Common abbreviations
    name_lower = name.lower()
    if 'partial least squares' in name_lower:
        variants.append(name_lower.replace('partial least squares', 'pls'))
        variants.append(name_lower.replace('partial least squares regression', 'plsr'))
    
    if 'principal component' in name_lower:
        variants.append(name_lower.replace('principal component analysis', 'pca'))
    
    # Remove common suffixes/prefixes
    variants.append(name.replace(' (PCA)', '').replace('(PCA)', ''))
    variants.append(name.replace(' (PLSR)', '').replace('(PLSR)', ''))
    
    return list(set(variants))  # Remove duplicates

