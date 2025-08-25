#!/usr/bin/env python3
"""
Скрипт для проверки архитектуры и оптимизации SAMe
"""

import ast
import importlib
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import warnings


class ArchitectureReviewer:
    """Класс для анализа архитектуры проекта"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_dir = project_root / "src"
        self.modules = ['same_core', 'same_clear', 'same_search', 'same_api', 'same']
        
    def check_circular_imports(self) -> Dict[str, List[str]]:
        """Проверка циркулярных импортов"""
        print("🔍 Проверка циркулярных импортов...")
        
        circular_imports = {}
        
        for module_name in self.modules:
            module_path = self.src_dir / module_name
            if not module_path.exists():
                continue
                
            try:
                # Очищаем кэш модулей
                modules_to_clear = [name for name in sys.modules.keys() if name.startswith(module_name)]
                for mod_name in modules_to_clear:
                    if mod_name in sys.modules:
                        del sys.modules[mod_name]
                
                # Пытаемся импортировать модуль
                module = importlib.import_module(module_name)
                print(f"✅ {module_name}: OK")
                
            except ImportError as e:
                if "circular import" in str(e).lower():
                    circular_imports[module_name] = [str(e)]
                    print(f"❌ {module_name}: Circular import detected - {e}")
                else:
                    print(f"⚠️  {module_name}: Import error - {e}")
        
        return circular_imports
    
    def analyze_import_dependencies(self) -> Dict[str, Dict[str, List[str]]]:
        """Анализ зависимостей между модулями"""
        print("\n📊 Анализ зависимостей между модулями...")
        
        dependencies = {}
        
        for module_name in self.modules:
            module_path = self.src_dir / module_name
            if not module_path.exists():
                continue
                
            dependencies[module_name] = {
                'imports': [],
                'imported_by': []
            }
            
            # Анализируем все Python файлы в модуле
            for py_file in module_path.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if any(alias.name.startswith(mod) for mod in self.modules):
                                    dependencies[module_name]['imports'].append(alias.name)
                        
                        elif isinstance(node, ast.ImportFrom):
                            if node.module and any(node.module.startswith(mod) for mod in self.modules):
                                dependencies[module_name]['imports'].append(node.module)
                
                except Exception as e:
                    print(f"⚠️  Ошибка анализа {py_file}: {e}")
        
        # Убираем дубликаты
        for module_name in dependencies:
            dependencies[module_name]['imports'] = list(set(dependencies[module_name]['imports']))
        
        # Вычисляем обратные зависимости
        for module_name, deps in dependencies.items():
            for imported_module in deps['imports']:
                base_module = imported_module.split('.')[0]
                if base_module in dependencies:
                    dependencies[base_module]['imported_by'].append(module_name)
        
        return dependencies
    
    def check_interface_compliance(self) -> Dict[str, List[str]]:
        """Проверка соответствия интерфейсам"""
        print("\n🔌 Проверка соответствия интерфейсам...")
        
        compliance_issues = {}
        
        try:
            from same_core.interfaces import (
                TextProcessorInterface,
                SearchEngineInterface,
                ExporterInterface
            )
            
            # Проверяем реализации в same_clear
            try:
                from same_clear.text_processing import TextCleaner
                
                issues = []
                if not hasattr(TextCleaner, 'clean_text'):
                    issues.append("TextCleaner missing clean_text method")
                
                if issues:
                    compliance_issues['same_clear.TextCleaner'] = issues
                else:
                    print("✅ same_clear.TextCleaner: соответствует интерфейсу")
                    
            except ImportError as e:
                compliance_issues['same_clear.TextCleaner'] = [f"Import error: {e}"]
            
            # Проверяем реализации в same_search
            try:
                from same_search.search_engine import FuzzySearchEngine
                
                issues = []
                required_methods = ['fit', 'search']
                for method in required_methods:
                    if not hasattr(FuzzySearchEngine, method):
                        issues.append(f"FuzzySearchEngine missing {method} method")
                
                if issues:
                    compliance_issues['same_search.FuzzySearchEngine'] = issues
                else:
                    print("✅ same_search.FuzzySearchEngine: соответствует интерфейсу")
                    
            except ImportError as e:
                compliance_issues['same_search.FuzzySearchEngine'] = [f"Import error: {e}"]
            
            # Проверяем реализации в same_api
            try:
                from same_api.export import ExcelExporter
                
                issues = []
                if not hasattr(ExcelExporter, 'export_data'):
                    issues.append("ExcelExporter missing export_data method")
                
                if issues:
                    compliance_issues['same_api.ExcelExporter'] = issues
                else:
                    print("✅ same_api.ExcelExporter: соответствует интерфейсу")
                    
            except ImportError as e:
                compliance_issues['same_api.ExcelExporter'] = [f"Import error: {e}"]
        
        except ImportError as e:
            compliance_issues['same_core.interfaces'] = [f"Core interfaces not available: {e}"]
        
        return compliance_issues
    
    def check_backward_compatibility(self) -> Dict[str, List[str]]:
        """Проверка обратной совместимости"""
        print("\n🔄 Проверка обратной совместимости...")
        
        compatibility_issues = {}
        
        # Тестируем старые импорты
        old_imports = [
            'same.text_processing.TextCleaner',
            'same.parameter_extraction.RegexParameterExtractor',
            'same.search_engine.FuzzySearchEngine',
            'same.export.ExcelExporter',
            'same.analog_search_engine.AnalogSearchEngine'
        ]
        
        for import_path in old_imports:
            try:
                module_path, class_name = import_path.rsplit('.', 1)
                
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    
                    module = importlib.import_module(module_path)
                    cls = getattr(module, class_name)
                    
                    if cls is None:
                        compatibility_issues[import_path] = ["Class not found"]
                    else:
                        # Проверяем что выдается предупреждение о deprecated
                        if w and any("deprecated" in str(warning.message).lower() for warning in w):
                            print(f"✅ {import_path}: работает с предупреждением")
                        else:
                            print(f"✅ {import_path}: работает")
                            
            except ImportError as e:
                compatibility_issues[import_path] = [f"Import failed: {e}"]
            except AttributeError as e:
                compatibility_issues[import_path] = [f"Attribute error: {e}"]
        
        return compatibility_issues
    
    def measure_performance(self) -> Dict[str, float]:
        """Измерение производительности импортов"""
        print("\n⚡ Измерение производительности импортов...")
        
        performance_metrics = {}
        
        # Тестируем время импорта модулей
        for module_name in self.modules:
            try:
                # Очищаем кэш
                modules_to_clear = [name for name in sys.modules.keys() if name.startswith(module_name)]
                for mod_name in modules_to_clear:
                    if mod_name in sys.modules:
                        del sys.modules[mod_name]
                
                # Измеряем время импорта
                start_time = time.time()
                importlib.import_module(module_name)
                import_time = time.time() - start_time
                
                performance_metrics[module_name] = import_time
                print(f"📊 {module_name}: {import_time:.3f}s")
                
            except ImportError as e:
                print(f"⚠️  {module_name}: не удалось импортировать - {e}")
        
        # Тестируем производительность обратной совместимости
        try:
            # Очищаем кэш
            modules_to_clear = [name for name in sys.modules.keys() if name.startswith('same')]
            for mod_name in modules_to_clear:
                if mod_name in sys.modules:
                    del sys.modules[mod_name]
            
            start_time = time.time()
            from same.text_processing import TextCleaner
            legacy_import_time = time.time() - start_time
            
            performance_metrics['legacy_import'] = legacy_import_time
            print(f"📊 legacy import: {legacy_import_time:.3f}s")
            
        except ImportError:
            print("⚠️  Legacy import не работает")
        
        return performance_metrics
    
    def check_code_quality(self) -> Dict[str, List[str]]:
        """Проверка качества кода"""
        print("\n🔍 Проверка качества кода...")
        
        quality_issues = {}
        
        for module_name in self.modules:
            module_path = self.src_dir / module_name
            if not module_path.exists():
                continue
            
            issues = []
            
            # Проверяем наличие __init__.py
            init_file = module_path / "__init__.py"
            if not init_file.exists():
                issues.append("Missing __init__.py")
            
            # Проверяем наличие docstrings в основных файлах
            for py_file in module_path.rglob("*.py"):
                if py_file.name.startswith('_'):
                    continue
                    
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    # Проверяем docstring модуля
                    if not ast.get_docstring(tree):
                        issues.append(f"Missing module docstring in {py_file.name}")
                    
                    # Проверяем docstrings классов
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            if not ast.get_docstring(node):
                                issues.append(f"Missing class docstring: {node.name} in {py_file.name}")
                
                except Exception as e:
                    issues.append(f"Error analyzing {py_file.name}: {e}")
            
            if issues:
                quality_issues[module_name] = issues
            else:
                print(f"✅ {module_name}: качество кода OK")
        
        return quality_issues
    
    def generate_report(self) -> str:
        """Генерация итогового отчета"""
        print("\n" + "="*60)
        print("📋 ОТЧЕТ ПО АРХИТЕКТУРЕ SAMe")
        print("="*60)
        
        # Запускаем все проверки
        circular_imports = self.check_circular_imports()
        dependencies = self.analyze_import_dependencies()
        interface_compliance = self.check_interface_compliance()
        backward_compatibility = self.check_backward_compatibility()
        performance = self.measure_performance()
        code_quality = self.check_code_quality()
        
        # Формируем отчет
        report = []
        report.append("# Отчет по архитектуре SAMe\n")
        
        # Циркулярные импорты
        if circular_imports:
            report.append("## ❌ Циркулярные импорты обнаружены:")
            for module, errors in circular_imports.items():
                report.append(f"- **{module}**: {', '.join(errors)}")
        else:
            report.append("## ✅ Циркулярные импорты: не обнаружены")
        
        report.append("")
        
        # Зависимости
        report.append("## 📊 Зависимости между модулями:")
        for module, deps in dependencies.items():
            if deps['imports']:
                report.append(f"- **{module}** импортирует: {', '.join(deps['imports'])}")
        
        report.append("")
        
        # Соответствие интерфейсам
        if interface_compliance:
            report.append("## ⚠️ Проблемы с интерфейсами:")
            for component, issues in interface_compliance.items():
                report.append(f"- **{component}**: {', '.join(issues)}")
        else:
            report.append("## ✅ Соответствие интерфейсам: OK")
        
        report.append("")
        
        # Обратная совместимость
        if backward_compatibility:
            report.append("## ⚠️ Проблемы с обратной совместимостью:")
            for import_path, issues in backward_compatibility.items():
                report.append(f"- **{import_path}**: {', '.join(issues)}")
        else:
            report.append("## ✅ Обратная совместимость: OK")
        
        report.append("")
        
        # Производительность
        report.append("## ⚡ Производительность импортов:")
        for module, time_taken in performance.items():
            status = "🟢" if time_taken < 1.0 else "🟡" if time_taken < 3.0 else "🔴"
            report.append(f"- {status} **{module}**: {time_taken:.3f}s")
        
        report.append("")
        
        # Качество кода
        if code_quality:
            report.append("## 📝 Проблемы качества кода:")
            for module, issues in code_quality.items():
                report.append(f"- **{module}**:")
                for issue in issues:
                    report.append(f"  - {issue}")
        else:
            report.append("## ✅ Качество кода: OK")
        
        report.append("")
        
        # Рекомендации
        report.append("## 💡 Рекомендации:")
        
        if circular_imports:
            report.append("- Устранить циркулярные импорты")
        
        if any(time_taken > 2.0 for time_taken in performance.values()):
            report.append("- Оптимизировать медленные импорты")
        
        if interface_compliance:
            report.append("- Привести компоненты в соответствие с интерфейсами")
        
        if backward_compatibility:
            report.append("- Исправить проблемы с обратной совместимостью")
        
        if not any([circular_imports, interface_compliance, backward_compatibility]):
            report.append("- Архитектура в хорошем состоянии! 🎉")
        
        return "\n".join(report)


def main():
    """Главная функция"""
    project_root = Path(__file__).parent.parent
    reviewer = ArchitectureReviewer(project_root)
    
    report = reviewer.generate_report()
    
    # Сохраняем отчет
    report_file = project_root / "docs" / "architecture_review.md"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 Отчет сохранен в: {report_file}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
