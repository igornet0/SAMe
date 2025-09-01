"""
Оптимизатор деревьев аналогов для устранения циклов и улучшения иерархии
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """Узел дерева аналогов"""
    index: int
    name: str
    level: int = 0
    children: List['TreeNode'] = None
    similarity_score: float = 0.0
    analog_type: str = "root"
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class TreeOptimizer:
    """Оптимизатор для создания иерархических деревьев без циклов"""
    
    def __init__(self, max_tree_depth: int = 5, min_similarity_for_parent: float = 0.1):
        """
        Args:
            max_tree_depth: Максимальная глубина дерева
            min_similarity_for_parent: Минимальная схожесть для создания родительской связи
        """
        self.max_tree_depth = max_tree_depth
        self.min_similarity_for_parent = min_similarity_for_parent
        
    def optimize_trees(self, analog_groups: List[Dict], processed_df: pd.DataFrame) -> List[Dict]:
        """
        Оптимизация деревьев аналогов с устранением циклов
        
        Args:
            analog_groups: Группы аналогов
            processed_df: DataFrame с обработанными данными
            
        Returns:
            Список оптимизированных деревьев
        """
        logger.info(f"Optimizing {len(analog_groups)} analog trees...")
        
        # Строим граф аналогов
        analog_graph = self._build_analog_graph(analog_groups)
        
        # Находим компоненты связности
        components = self._find_connected_components(analog_graph)
        
        # Оптимизируем каждую компоненту
        optimized_trees = []
        for i, component in enumerate(components):
            logger.debug(f"Processing component {i+1}/{len(components)} with {len(component)} nodes")
            
            # Создаем иерархическое дерево для компоненты
            tree = self._create_hierarchical_tree(component, analog_graph, processed_df)
            if tree:
                optimized_trees.append(tree)
        
        logger.info(f"Created {len(optimized_trees)} optimized trees")
        return optimized_trees
    
    def _build_analog_graph(self, analog_groups: List[Dict]) -> Dict[int, Dict[int, float]]:
        """Построение графа аналогов"""
        graph = defaultdict(dict)
        
        for group in analog_groups:
            if isinstance(group, dict):
                main_index = group.get('main_index', 0)
                analogs = group.get('analogs', [])
            else:
                main_index = group.reference_index
                analogs = group.analogs
            
            for analog in analogs:
                if isinstance(analog, dict):
                    analog_index = analog.get('similar_index', analog.get('index', 0))
                    similarity = analog.get('similarity_score', analog.get('similarity', 0.0))
                else:
                    analog_index = analog.index
                    similarity = analog.similarity
                
                # Добавляем связь в граф (неориентированный)
                graph[main_index][analog_index] = similarity
                graph[analog_index][main_index] = similarity
        
        return dict(graph)
    
    def _find_connected_components(self, graph: Dict[int, Dict[int, float]]) -> List[Set[int]]:
        """Поиск компонент связности в графе"""
        visited = set()
        components = []
        
        for node in graph.keys():
            if node not in visited:
                component = set()
                queue = deque([node])
                
                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        
                        # Добавляем соседей
                        for neighbor in graph.get(current, {}):
                            if neighbor not in visited:
                                queue.append(neighbor)
                
                if component:
                    components.append(component)
        
        return components
    
    def _create_hierarchical_tree(self, component: Set[int], graph: Dict[int, Dict[int, float]], 
                                processed_df: pd.DataFrame) -> Optional[Dict]:
        """Создание иерархического дерева для компоненты"""
        if len(component) < 2:
            return None
        
        # Выбираем корень - узел с наибольшим количеством связей и наибольшей суммарной схожестью
        root_node = self._select_root_node(component, graph)
        
        # Строим дерево с помощью BFS для избежания циклов
        tree = self._build_tree_bfs(root_node, component, graph, processed_df)
        
        return tree
    
    def _select_root_node(self, component: Set[int], graph: Dict[int, Dict[int, float]]) -> int:
        """Выбор корневого узла для дерева"""
        best_node = None
        best_score = -1
        
        for node in component:
            # Считаем метрику для выбора корня
            connections = len(graph.get(node, {}))
            avg_similarity = np.mean(list(graph.get(node, {}).values())) if connections > 0 else 0
            total_similarity = sum(graph.get(node, {}).values())
            
            # Комбинированная метрика: количество связей + средняя схожесть + общая схожесть
            score = connections * 0.4 + avg_similarity * 0.3 + total_similarity * 0.3
            
            if score > best_score:
                best_score = score
                best_node = node
        
        return best_node or list(component)[0]
    
    def _build_tree_bfs(self, root: int, component: Set[int], graph: Dict[int, Dict[int, float]], 
                       processed_df: pd.DataFrame) -> Dict:
        """Построение дерева с помощью BFS для избежания циклов"""
        
        # Получаем название корневого узла
        name_column = self._get_name_column(processed_df)
        root_name = processed_df.loc[root, name_column] if root in processed_df.index else f"Item_{root}"
        
        tree = {
            'root_index': root,
            'root_name': root_name,
            'exact_analogs': [],
            'close_analogs': [],
            'possible_analogs': [],
            'total_nodes': len(component),
            'tree_depth': 0
        }
        
        visited = {root}
        queue = deque([(root, 0)])  # (node, level)
        max_depth = 0
        
        while queue and max_depth < self.max_tree_depth:
            current_node, level = queue.popleft()
            max_depth = max(max_depth, level)
            
            # Получаем соседей текущего узла
            neighbors = graph.get(current_node, {})
            
            # Сортируем соседей по схожести (убывание)
            sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
            
            for neighbor, similarity in sorted_neighbors:
                if neighbor not in visited and neighbor in component:
                    visited.add(neighbor)
                    
                    # Получаем название соседа
                    neighbor_name = processed_df.loc[neighbor, name_column] if neighbor in processed_df.index else f"Item_{neighbor}"
                    
                    # Классифицируем тип аналога
                    analog_data = {
                        'index': neighbor,
                        'name': neighbor_name,
                        'similarity': similarity,
                        'level': level + 1
                    }
                    
                    if similarity >= 0.7:
                        tree['exact_analogs'].append(analog_data)
                    elif similarity >= 0.5:
                        tree['close_analogs'].append(analog_data)
                    else:
                        tree['possible_analogs'].append(analog_data)
                    
                    # Добавляем в очередь для дальнейшего обхода
                    if level + 1 < self.max_tree_depth:
                        queue.append((neighbor, level + 1))
        
        tree['tree_depth'] = max_depth
        
        # Ограничиваем количество аналогов каждого типа
        tree['exact_analogs'] = tree['exact_analogs'][:10]
        tree['close_analogs'] = tree['close_analogs'][:15]
        tree['possible_analogs'] = tree['possible_analogs'][:20]
        
        return tree
    
    def _get_name_column(self, df: pd.DataFrame) -> str:
        """Определение колонки с названиями товаров"""
        for col in df.columns:
            if 'наименование' in col.lower() or 'название' in col.lower() or 'name' in col.lower():
                return col
        return df.columns[0] if len(df.columns) > 0 else 'index'
    
    def create_graph_analysis(self, analog_groups: List[Dict]) -> Dict[str, Any]:
        """Анализ графа аналогов"""
        graph = self._build_analog_graph(analog_groups)
        components = self._find_connected_components(graph)
        
        # Статистика по компонентам
        component_sizes = [len(comp) for comp in components]
        
        # Статистика по связям
        total_edges = sum(len(neighbors) for neighbors in graph.values()) // 2  # Неориентированный граф
        
        # Поиск циклов
        cycles_count = self._count_cycles(graph)
        
        analysis = {
            'total_nodes': len(graph),
            'total_edges': total_edges,
            'connected_components': len(components),
            'largest_component_size': max(component_sizes) if component_sizes else 0,
            'average_component_size': np.mean(component_sizes) if component_sizes else 0,
            'components_with_cycles': cycles_count,
            'density': total_edges / (len(graph) * (len(graph) - 1) / 2) if len(graph) > 1 else 0
        }
        
        return analysis
    
    def _count_cycles(self, graph: Dict[int, Dict[int, float]]) -> int:
        """Подсчет количества циклов в графе"""
        try:
            # Используем NetworkX для анализа циклов
            G = nx.Graph()
            for node, neighbors in graph.items():
                for neighbor, weight in neighbors.items():
                    G.add_edge(node, neighbor, weight=weight)
            
            # Подсчитываем циклы длиной 3 (треугольники)
            triangles = sum(1 for _ in nx.enumerate_all_cliques(G) if len(list(_)) == 3)
            return triangles
            
        except ImportError:
            # Простой подсчет без NetworkX
            cycles = 0
            visited = set()
            
            for node in graph:
                if node not in visited:
                    cycles += self._dfs_cycle_count(node, -1, graph, visited, set())
            
            return cycles
    
    def _dfs_cycle_count(self, node: int, parent: int, graph: Dict[int, Dict[int, float]], 
                        global_visited: Set[int], local_visited: Set[int]) -> int:
        """DFS для подсчета циклов"""
        local_visited.add(node)
        cycles = 0
        
        for neighbor in graph.get(node, {}):
            if neighbor == parent:
                continue
                
            if neighbor in local_visited:
                cycles += 1
            elif neighbor not in global_visited:
                cycles += self._dfs_cycle_count(neighbor, node, graph, global_visited, local_visited)
        
        global_visited.add(node)
        return cycles
