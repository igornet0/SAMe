#!/usr/bin/env python3
"""
Hierarchy Tree Builder для построения иерархического дерева аналогов и дублей
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TreeBuilderConfig:
    """Конфигурация построения дерева"""
    min_similarity_for_tree: float = 0.5
    max_tree_depth: int = 5
    sort_by_similarity: bool = True
    include_noise_points: bool = True
    representative_selection: str = "shortest"  # shortest, most_similar, random
    max_children_per_node: int = 20


@dataclass
class HierarchyNode:
    """Узел иерархического дерева"""
    code: str
    name: str
    similarity: Optional[float] = None
    relation_type: str = "unknown"
    children: List['HierarchyNode'] = field(default_factory=list)
    parent: Optional['HierarchyNode'] = None
    depth: int = 0
    cluster_id: Optional[int] = None
    is_noise: bool = False
    
    def add_child(self, child: 'HierarchyNode'):
        """Добавление дочернего узла"""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
    
    def get_all_children(self) -> List['HierarchyNode']:
        """Получение всех дочерних узлов (рекурсивно)"""
        all_children = []
        for child in self.children:
            all_children.append(child)
            all_children.extend(child.get_all_children())
        return all_children
    
    def get_tree_size(self) -> int:
        """Получение размера поддерева"""
        return 1 + sum(child.get_tree_size() for child in self.children)


class SimilarityCalculator:
    """Калькулятор схожести для определения типов отношений"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.thresholds = {
            'duplicate': 0.95,
            'analog': 0.8,
            'close_analog': 0.7,
            'possible_analog': 0.6,
            'similar_product': 0.5
        }
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Вычисление схожести между двумя текстами"""
        if not text1 or not text2:
            return 0.0
        
        # Простая схожесть по словам
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def determine_relation_type(self, similarity: float) -> str:
        """Определение типа отношения на основе схожести"""
        if similarity >= self.thresholds['duplicate']:
            return "дубль"
        elif similarity >= self.thresholds['analog']:
            return "аналог"
        elif similarity >= self.thresholds['close_analog']:
            return "близкий аналог"
        elif similarity >= self.thresholds['possible_analog']:
            return "возможный аналог"
        elif similarity >= self.thresholds['similar_product']:
            return "похожий товар"
        else:
            return "нет аналогов"


class HierarchyTreeBuilder:
    """Построитель иерархического дерева аналогов"""
    
    def __init__(self, config: TreeBuilderConfig = None):
        self.config = config or TreeBuilderConfig()
        self.similarity_calculator = SimilarityCalculator()
        self.root_nodes = []
    
    def build_hierarchy_tree(self, catalog_df: pd.DataFrame, clusters: Dict, 
                           noise_assignments: Dict) -> List[HierarchyNode]:
        """Построение иерархического дерева аналогов"""
        
        logger.info("Building hierarchy tree...")
        
        # 1. Создаем корневые узлы для каждого кластера
        self._create_cluster_roots(catalog_df, clusters)
        
        # 2. Обрабатываем шумовые точки
        if self.config.include_noise_points:
            self._create_noise_roots(catalog_df, noise_assignments)
        
        # 3. Сортируем узлы
        if self.config.sort_by_similarity:
            self._sort_nodes_by_similarity()
        
        logger.info(f"Hierarchy tree built with {len(self.root_nodes)} root nodes")
        return self.root_nodes
    
    def _create_cluster_roots(self, catalog_df: pd.DataFrame, clusters: Dict):
        """Создание корневых узлов для кластеров"""
        for cluster_id, cluster_indices in clusters.items():
            if cluster_id == -1:  # Шумовые точки обрабатываем отдельно
                continue
            
            # Находим представителя кластера
            representative_idx = self._find_cluster_representative(cluster_indices, catalog_df)
            
            # Создаем корневой узел
            root_node = HierarchyNode(
                code=catalog_df.iloc[representative_idx]['Код'],
                name=catalog_df.iloc[representative_idx]['Наименование'],
                similarity=None,
                relation_type="root",
                cluster_id=cluster_id,
                is_noise=False
            )
            
            # Добавляем все записи кластера как дочерние узлы
            for idx in cluster_indices:
                if idx != representative_idx:
                    similarity = self.similarity_calculator.calculate_similarity(
                        catalog_df.iloc[representative_idx]['Наименование'],
                        catalog_df.iloc[idx]['Наименование']
                    )
                    
                    if similarity >= self.config.min_similarity_for_tree:
                        child_node = HierarchyNode(
                            code=catalog_df.iloc[idx]['Код'],
                            name=catalog_df.iloc[idx]['Наименование'],
                            similarity=similarity,
                            relation_type=self.similarity_calculator.determine_relation_type(similarity),
                            cluster_id=cluster_id,
                            is_noise=False
                        )
                        root_node.add_child(child_node)
            
            self.root_nodes.append(root_node)
    
    def _create_noise_roots(self, catalog_df: pd.DataFrame, noise_assignments: Dict):
        """Создание корневых узлов для шумовых точек"""
        for noise_idx, analogs in noise_assignments.items():
            # Создаем корневой узел для шумовой точки
            noise_node = HierarchyNode(
                code=catalog_df.iloc[noise_idx]['Код'],
                name=catalog_df.iloc[noise_idx]['Наименование'],
                similarity=None,
                relation_type="noise",
                cluster_id=-1,
                is_noise=True
            )
            
            # Добавляем аналогов как дочерние узлы
            for analog_idx, similarity in analogs:
                if similarity >= self.config.min_similarity_for_tree:
                    analog_node = HierarchyNode(
                        code=catalog_df.iloc[analog_idx]['Код'],
                        name=catalog_df.iloc[analog_idx]['Наименование'],
                        similarity=similarity,
                        relation_type=self.similarity_calculator.determine_relation_type(similarity),
                        cluster_id=catalog_df.iloc[analog_idx].get('cluster_label', -1),
                        is_noise=False
                    )
                    noise_node.add_child(analog_node)
            
            # Добавляем узел только если у него есть дети
            if noise_node.children:
                self.root_nodes.append(noise_node)
    
    def _find_cluster_representative(self, cluster_indices: List[int], 
                                   catalog_df: pd.DataFrame) -> int:
        """Находит представителя кластера"""
        if self.config.representative_selection == "shortest":
            return self._find_shortest_name(cluster_indices, catalog_df)
        elif self.config.representative_selection == "most_similar":
            return self._find_most_similar(cluster_indices, catalog_df)
        else:
            return cluster_indices[0]  # Первый элемент
    
    def _find_shortest_name(self, cluster_indices: List[int], 
                          catalog_df: pd.DataFrame) -> int:
        """Находит запись с самым коротким наименованием"""
        shortest_idx = cluster_indices[0]
        shortest_length = len(catalog_df.iloc[shortest_idx]['Наименование'])
        
        for idx in cluster_indices[1:]:
            name_length = len(catalog_df.iloc[idx]['Наименование'])
            if name_length < shortest_length:
                shortest_length = name_length
                shortest_idx = idx
        
        return shortest_idx
    
    def _find_most_similar(self, cluster_indices: List[int], 
                         catalog_df: pd.DataFrame) -> int:
        """Находит запись с наибольшей средней схожестью с остальными"""
        best_idx = cluster_indices[0]
        best_avg_similarity = 0.0
        
        for idx in cluster_indices:
            similarities = []
            for other_idx in cluster_indices:
                if idx != other_idx:
                    similarity = self.similarity_calculator.calculate_similarity(
                        catalog_df.iloc[idx]['Наименование'],
                        catalog_df.iloc[other_idx]['Наименование']
                    )
                    similarities.append(similarity)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                if avg_similarity > best_avg_similarity:
                    best_avg_similarity = avg_similarity
                    best_idx = idx
        
        return best_idx
    
    def _sort_nodes_by_similarity(self):
        """Сортировка узлов по схожести"""
        for root_node in self.root_nodes:
            # Сортируем дочерние узлы по убыванию схожести
            root_node.children.sort(key=lambda x: x.similarity or 0, reverse=True)
            
            # Ограничиваем количество дочерних узлов
            if len(root_node.children) > self.config.max_children_per_node:
                root_node.children = root_node.children[:self.config.max_children_per_node]
    
    def generate_tree_text(self, nodes: List[HierarchyNode] = None, level: int = 0) -> str:
        """Генерация текстового представления дерева"""
        if nodes is None:
            nodes = self.root_nodes
        
        tree_text = ""
        indent = "    " * level
        
        for node in nodes:
            # Формируем строку узла
            if node.similarity is not None:
                node_line = f"{indent}- {node.code} | {node.name} [{node.relation_type}] ({node.similarity:.4f})"
            else:
                node_line = f"{indent}- {node.code} | {node.name} ({node.relation_type})"
            
            tree_text += node_line + "\n"
            
            # Рекурсивно обрабатываем дочерние узлы
            if node.children and level < self.config.max_tree_depth:
                tree_text += self.generate_tree_text(node.children, level + 1)
        
        return tree_text
    
    def save_tree_to_file(self, output_path: str, nodes: List[HierarchyNode] = None) -> str:
        """Сохранение дерева в файл"""
        tree_text = self.generate_tree_text(nodes)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(tree_text)
        
        logger.info(f"Tree saved to: {output_path}")
        return output_path
    
    def get_tree_statistics(self) -> Dict[str, Any]:
        """Получение статистики дерева"""
        total_nodes = 0
        total_relations = 0
        relation_types = {}
        max_depth = 0
        
        def traverse_node(node: HierarchyNode, depth: int):
            nonlocal total_nodes, total_relations, max_depth
            
            total_nodes += 1
            max_depth = max(max_depth, depth)
            
            if node.similarity is not None:
                total_relations += 1
                relation_type = node.relation_type
                relation_types[relation_type] = relation_types.get(relation_type, 0) + 1
            
            for child in node.children:
                traverse_node(child, depth + 1)
        
        for root_node in self.root_nodes:
            traverse_node(root_node, 0)
        
        return {
            'total_nodes': total_nodes,
            'total_relations': total_relations,
            'max_depth': max_depth,
            'relation_types': relation_types,
            'root_nodes': len(self.root_nodes)
        }

