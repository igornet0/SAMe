#!/usr/bin/env python3
"""
Search Engine модуль для поиска аналогов и построения иерархических деревьев
"""

from .enhanced_dbscan_search import (
    EnhancedDBSCANSearch,
    EnhancedDBSCANConfig,
    NoiseProcessorConfig,
    RelationType
)

from .hierarchy_tree_builder import (
    HierarchyTreeBuilder,
    TreeBuilderConfig,
    HierarchyNode,
    SimilarityCalculator
)

from .tree_generator import (
    TreeGenerator,
    TreeGeneratorConfig,
    generate_search_trees_from_file
)

__all__ = [
    # Enhanced DBSCAN Search
    'EnhancedDBSCANSearch',
    'EnhancedDBSCANConfig', 
    'NoiseProcessorConfig',
    'RelationType',
    
    # Hierarchy Tree Builder
    'HierarchyTreeBuilder',
    'TreeBuilderConfig',
    'HierarchyNode',
    'SimilarityCalculator',
    
    # Tree Generator
    'TreeGenerator',
    'TreeGeneratorConfig',
    'generate_search_trees_from_file'
]

__version__ = "1.0.0"
__author__ = "SAMe Team"

