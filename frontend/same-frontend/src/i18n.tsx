import React, { createContext, useContext, useMemo, useState, useEffect, ReactNode } from 'react';

type Language = 'en' | 'ru';

type TranslationDict = Record<string, string>;

type Translations = Record<Language, TranslationDict>;

const translations: Translations = {
  en: {
    'common.connected': 'Connected',
    'common.tryAgain': 'Try Again',
    'common.remove': 'Remove',
    'common.previous': 'Previous',
    'common.next': 'Next',
    'common.value': 'Value',
    'common.count': 'Count',
    'common.percent': '%',
    'common.code': 'Code',
    'common.name': 'Name',
    'common.error': 'Error',
    'common.null': 'NULL',
    'common.records': 'records',

    'app.title': 'SAMe - Analog Search Engine',
    'app.subtitle': 'Search for material and technical resource analogs',
    'app.footer': 'SAMe v1.2.0 - Search Analog Model Engine',
    'app.connecting': 'Connecting to SAMe backend...',
    'app.connectedToast.title': 'Connected',
    'app.connectedToast.message': 'Connected to SAMe backend successfully',
    'app.connectError.title': 'Backend Connection Failed',
    'app.connectError.message': 'Unable to connect to the SAMe backend server. Please ensure the server is running on http://localhost:8000',

    'upload.title': 'Upload Catalog File',
    'upload.subtitle': 'Upload your catalog file in CSV, Excel, or JSON format',
    'upload.dropHere': 'Drop your file here, or click to browse',
    'upload.dropNow': 'Drop the file here',
    'upload.supports': 'Supports CSV, Excel (.xlsx, .xls), and JSON files up to 50MB',
    'upload.aria': 'Upload catalog file',
    'upload.inputAria': 'File upload input',
    'upload.uploading': 'Uploading...',
    'upload.processing': 'Processing...',
    'upload.processingHint': 'This may take several minutes for large files',
    'upload.validation.type': 'Only CSV, Excel (.xlsx, .xls), and JSON files are supported',
    'upload.validation.size': 'File size must be less than 50MB',
    'upload.successToast.title': 'Upload Complete',
    'upload.successToast.message': 'Catalog uploaded successfully! {count}',
    'upload.successToast.count': '{count} records loaded.',
    'upload.errorToast.title': 'Upload Failed',
    'upload.partialSuccessToast.title': 'Uploaded with issues',
    'upload.partialSuccessToast.message': 'Catalog uploaded, but some rows failed to process. {count}',

    'search.title': 'Search for Analogs',
    'search.subtitle': 'Enter a product name to find similar items in your catalog',
    'search.alert.needUpload': 'Please upload a catalog file first before searching for analogs',
    'search.productName': 'Product Name',
    'search.placeholder': "Enter product name (e.g., 'Болт М10х50')",
    'search.helper': 'Enter the name of the product you want to find analogs for',
    'search.method': 'Search Method',
    'search.method.hybrid': 'Hybrid (Recommended)',
    'search.method.semantic': 'Semantic Search',
    'search.method.fuzzy': 'Fuzzy Search',
    'search.method.token': 'Token Search',
    'search.method.hybrid_dbscan': 'Hybrid DBSCAN',
    'search.method.optimized_dbscan': 'Optimized DBSCAN',
    'search.method.help': 'Hybrid search combines semantic and fuzzy matching for best results. Token/DBSCAN methods require backend support.',
    'search.button': 'Search Analogs',
    'search.button.loading': 'Searching...',
    'search.loading': 'Searching for analogs...',
    'search.loadingHint': 'This may take a few moments depending on catalog size',
    'search.validation.empty': 'Please enter a product name to search for',
    'search.validation.short': 'Search query must be at least 2 characters long',
    'search.errorToast.title': 'Search Failed',
    'search.noResults.title': 'No results found',
    'search.noResults.text': 'No analogs were found for "{query}". Try adjusting your search terms or using a different search method.',
    'search.results.title': 'Search Results for "{query}"',
    'search.results.count': 'Found {count} analog{plural}',
    'search.results.export': 'Export to Excel',
    'search.table.empty': 'No results to display',
    'search.method.badge.semantic': 'Semantic',
    'search.method.badge.hybrid': 'Hybrid',
    'search.method.badge.fuzzy': 'Fuzzy',
    'search.method.badge.unknown': 'Unknown',

    'results.rank': 'Rank',
    'results.foundAnalog': 'Found Analog',
    'results.similarity': 'Similarity',
    'results.method': 'Method',
    'results.id': 'ID',
    'results.showing': 'Showing {from} to {to} of {total} results',

    'dataset.title': 'Dataset Statistics',
    'dataset.totalRows': 'Total rows: {total} • Columns: {cols}',
    'dataset.columns': 'Columns',
    'dataset.missing': '{count} missing ({pct}%)',
    'dataset.columnDetails': 'Column details',
    'dataset.dtype': 'dtype',
    'dataset.unique': 'unique',
    'dataset.missingShort': 'missing',
    'dataset.filterLabel': 'Filter by value (top 50):',
    'dataset.selectValue': 'Select a value...',
    'dataset.rowsWithValue': 'Rows with value "{value}":',
    'dataset.processingReport': 'Processing Report',
    'dataset.failedRows': 'Failed rows: {count}',
    'dataset.showFailed': 'Show failed examples ({count} shown)',
    'dataset.failedHint': 'Some rows could not be processed. Search will continue using the successfully processed data.'
  },
  ru: {
    'common.connected': 'Подключено',
    'common.tryAgain': 'Повторить',
    'common.remove': 'Удалить',
    'common.previous': 'Назад',
    'common.next': 'Вперёд',
    'common.value': 'Значение',
    'common.count': 'Количество',
    'common.percent': '%',
    'common.code': 'Код',
    'common.name': 'Наименование',
    'common.error': 'Ошибка',
    'common.null': 'NULL',
    'common.records': 'записей',

    'app.title': 'SAMe — Поисковый движок аналогов',
    'app.subtitle': 'Поиск аналогов материалов и технических ресурсов',
    'app.footer': 'SAMe v1.2.0 — Search Analog Model Engine',
    'app.connecting': 'Подключение к серверу SAMe...',
    'app.connectedToast.title': 'Подключено',
    'app.connectedToast.message': 'Успешно подключено к бэкенду SAMe',
    'app.connectError.title': 'Ошибка подключения к бэкенду',
    'app.connectError.message': 'Не удалось подключиться к серверу SAMe. Убедитесь, что сервер запущен на http://localhost:8000',

    'upload.title': 'Загрузка файла каталога',
    'upload.subtitle': 'Загрузите файл каталога в формате CSV, Excel или JSON',
    'upload.dropHere': 'Перетащите файл сюда или нажмите для выбора',
    'upload.dropNow': 'Отпустите файл здесь',
    'upload.supports': 'Поддерживаются CSV, Excel (.xlsx, .xls) и JSON файлы до 50 МБ',
    'upload.aria': 'Загрузить файл каталога',
    'upload.inputAria': 'Поле загрузки файла',
    'upload.uploading': 'Загрузка...',
    'upload.processing': 'Обработка...',
    'upload.processingHint': 'Для больших файлов это может занять несколько минут',
    'upload.validation.type': 'Поддерживаются только файлы CSV, Excel (.xlsx, .xls) и JSON',
    'upload.validation.size': 'Размер файла должен быть меньше 50 МБ',
    'upload.successToast.title': 'Загрузка завершена',
    'upload.successToast.message': 'Каталог успешно загружен! {count}',
    'upload.successToast.count': 'Загружено записей: {count}.',
    'upload.errorToast.title': 'Ошибка загрузки',
    'upload.partialSuccessToast.title': 'Загружено с ошибками',
    'upload.partialSuccessToast.message': 'Каталог загружен, но некоторые строки не обработаны. {count}',

    'search.title': 'Поиск аналогов',
    'search.subtitle': 'Введите наименование, чтобы найти похожие позиции в каталоге',
    'search.alert.needUpload': 'Сначала загрузите файл каталога, прежде чем искать аналоги',
    'search.productName': 'Наименование',
    'search.placeholder': "Введите наименование (например, 'Болт М10х50')",
    'search.helper': 'Введите наименование позиции, для которой нужно найти аналоги',
    'search.method': 'Метод поиска',
    'search.method.hybrid': 'Гибридный (рекомендуется)',
    'search.method.semantic': 'Семантический поиск',
    'search.method.fuzzy': 'Нечёткий поиск',
    'search.method.token': 'Токен-поиск',
    'search.method.hybrid_dbscan': 'Гибридный DBSCAN',
    'search.method.optimized_dbscan': 'Оптимизированный DBSCAN',
    'search.method.help': 'Гибридный поиск сочетает семантическое и нечеткое сопоставление. Методы Token/DBSCAN требуют поддержку на сервере.',
    'search.button': 'Найти аналоги',
    'search.button.loading': 'Поиск...',
    'search.loading': 'Выполняется поиск аналогов...',
    'search.loadingHint': 'Это может занять время в зависимости от размера каталога',
    'search.validation.empty': 'Введите наименование для поиска',
    'search.validation.short': 'Запрос должен быть не короче 2 символов',
    'search.errorToast.title': 'Ошибка поиска',
    'search.noResults.title': 'Ничего не найдено',
    'search.noResults.text': 'Аналоги для "{query}" не найдены. Попробуйте изменить запрос или метод поиска.',
    'search.results.title': 'Результаты поиска по запросу "{query}"',
    'search.results.count': 'Найдено {count} аналог{plural}',
    'search.results.export': 'Экспорт в Excel',
    'search.table.empty': 'Нет данных для отображения',
    'search.method.badge.semantic': 'Semantic',
    'search.method.badge.hybrid': 'Hybrid',
    'search.method.badge.fuzzy': 'Fuzzy',
    'search.method.badge.unknown': 'Unknown',

    'results.rank': 'Рейтинг',
    'results.foundAnalog': 'Найденный аналог',
    'results.similarity': 'Схожесть',
    'results.method': 'Метод',
    'results.id': 'ID',
    'results.showing': 'Показаны {from}–{to} из {total} результатов',

    'dataset.title': 'Статистика датасета',
    'dataset.totalRows': 'Всего строк: {total} • Колонок: {cols}',
    'dataset.columns': 'Колонки',
    'dataset.missing': '{count} пропусков ({pct}%)',
    'dataset.columnDetails': 'Детали колонки',
    'dataset.dtype': 'тип',
    'dataset.unique': 'уникальных',
    'dataset.missingShort': 'пропусков',
    'dataset.filterLabel': 'Фильтр по значению (топ-50):',
    'dataset.selectValue': 'Выберите значение...',
    'dataset.rowsWithValue': 'Строк со значением "{value}":',
    'dataset.processingReport': 'Отчёт по обработке',
    'dataset.failedRows': 'Неуспешные строки: {count}',
    'dataset.showFailed': 'Показать примеры ошибок (показано {count})',
    'dataset.failedHint': 'Часть строк не удалось обработать. Поиск продолжится по успешно обработанным данным.'
  }
};

function interpolate(template: string, params?: Record<string, string | number>): string {
  if (!params) return template;
  return template.replace(/\{(\w+)\}/g, (_, k) => String(params[k] ?? ''));
}

interface I18nContextValue {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string, params?: Record<string, string | number>) => string;
}

const I18nContext = createContext<I18nContextValue | undefined>(undefined);

export const I18nProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const detectDefaultLanguage = (): Language => {
    if (typeof process !== 'undefined' && process.env && process.env.NODE_ENV === 'test') {
      return 'en';
    }
    try {
      const stored = typeof window !== 'undefined' ? window.localStorage.getItem('same_language') as Language | null : null;
      if (stored === 'en' || stored === 'ru') return stored;
    } catch {}
    if (typeof navigator !== 'undefined') {
      const nav = (navigator.language || '').toLowerCase();
      if (nav.startsWith('ru')) return 'ru';
    }
    return 'ru';
  };

  const [language, setLanguage] = useState<Language>(detectDefaultLanguage);

  useEffect(() => {
    try {
      if (typeof window !== 'undefined') {
        window.localStorage.setItem('same_language', language);
      }
    } catch {}
  }, [language]);

  const t = useMemo(() => {
    return (key: string, params?: Record<string, string | number>) => {
      const dict = translations[language] || translations.en;
      const template = dict[key] ?? translations.en[key] ?? key;
      return interpolate(template, params);
    };
  }, [language]);

  const value: I18nContextValue = useMemo(() => ({ language, setLanguage, t }), [language, t]);

  return (
    <I18nContext.Provider value={value}>
      {children}
    </I18nContext.Provider>
  );
};

export function useI18n(): I18nContextValue {
  const ctx = useContext(I18nContext);
  if (!ctx) {
    if (typeof process !== 'undefined' && process.env && process.env.NODE_ENV === 'test') {
      const fallbackT = (key: string, params?: Record<string, string | number>) => {
        const template = translations.en[key] ?? key;
        return interpolate(template, params);
      };
      return { language: 'en', setLanguage: () => {}, t: fallbackT } as I18nContextValue;
    }
    throw new Error('useI18n must be used within I18nProvider');
  }
  return ctx;
}

export const LanguageSwitcher: React.FC<{ className?: string }> = ({ className = '' }) => {
  const { language, setLanguage, t } = useI18n();
  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      <label htmlFor="lang-select" className="text-sm text-gray-600">{language === 'ru' ? 'Язык' : 'Language'}</label>
      <select
        id="lang-select"
        value={language}
        onChange={(e) => setLanguage(e.target.value as Language)}
        className="block px-2 py-1 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        aria-label={language === 'ru' ? 'Выбор языка' : 'Select language'}
      >
        <option value="ru">Русский</option>
        <option value="en">English</option>
      </select>
    </div>
  );
};


