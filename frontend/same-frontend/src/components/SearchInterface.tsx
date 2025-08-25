import React, { useState } from 'react';
import { Search } from 'lucide-react';
import { Button, Input, Alert, LoadingSpinner } from './ui';
import { SearchResult, SearchResponse, SearchRequest } from '../types/api';
import { useI18n } from '../i18n';

interface SearchInterfaceProps {
  catalogUploaded: boolean;
  onSearchResults: (results: SearchResult[], query: string) => void;
  onSearchError: (error: string) => void;
  disabled?: boolean;
}

const SearchInterface: React.FC<SearchInterfaceProps> = ({
  catalogUploaded,
  onSearchResults,
  onSearchError,
  disabled = false
}) => {
  const { t } = useI18n();
  const [query, setQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchMethod, setSearchMethod] = useState<string>('hybrid');

  const validateQuery = (searchQuery: string): string | null => {
    if (!searchQuery.trim()) {
      return t('search.validation.empty');
    }
    
    if (searchQuery.trim().length < 2) {
      return t('search.validation.short');
    }
    
    return null;
  };

  const performSearch = async () => {

    const validationError = validateQuery(query);
    if (validationError) {
      setError(validationError);
      onSearchError(validationError);
      return;
    }

    setSearching(true);
    setError(null);

    try {
      const searchRequest: SearchRequest = {
        queries: [query.trim()],
        method: searchMethod,
        similarity_threshold: 0.6,
        max_results: 10
      };

      const response = await fetch('http://localhost:8000/search/search-analogs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(searchRequest),
      });

      if (!response.ok) {
        // Автоинициализация при отсутствии движка
        const errorData = await response.json();
        if (errorData?.detail === 'Search engine is not initialized') {
          // Пытаемся инициализировать движок по тестовому датасету (или ранее загруженному файлу)
          const initResp = await fetch('http://localhost:8000/search/initialize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              catalog_file_path: 'src/data/input/test_catalog_10000.csv',
              search_method: searchMethod,
              similarity_threshold: 0.6
            })
          });
          if (!initResp.ok) {
            const initErr = await initResp.json().catch(() => ({}));
            throw new Error(initErr.detail || 'Failed to initialize search engine');
          }
          // Повторяем запрос поиска после успешной инициализации
          const retry = await fetch('http://localhost:8000/search/search-analogs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(searchRequest),
          });
          if (!retry.ok) {
            const retryErr = await retry.json().catch(() => ({}));
            throw new Error(retryErr.detail || 'Search failed after initialization');
          }
          const retrResult: SearchResponse = await retry.json();
          const retrQueryResults = retrResult.results[query.trim()] || [];
          onSearchResults(retrQueryResults, query.trim());
          return;
        }
        throw new Error(errorData.detail || 'Search failed');
      }

      const result: SearchResponse = await response.json();
      const queryResults = result.results[query.trim()] || [];
      
      onSearchResults(queryResults, query.trim());

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Search failed';
      setError(errorMessage);
      onSearchError(errorMessage);
    } finally {
      setSearching(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    performSearch();
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !searching) {
      performSearch();
    }
  };

  return (
    <div className="space-y-4">
      <div className="text-center">
        <h2 className="text-lg font-semibold text-gray-900 mb-2">
          Search for Analogs
        </h2>
        <p className="text-sm text-gray-600 mb-4">
          Enter a product name to find similar items in your catalog
        </p>
      </div>

      {error && (
        <Alert
          type="error"
          message={error}
          onClose={() => setError(null)}
        />
      )}

      {!catalogUploaded && (
        <Alert
          type="warning"
          message={t('search.alert.needUpload')}
        />
      )}

      <form onSubmit={handleSubmit} className="space-y-4 sm:space-y-6">
        <div className="space-y-4 sm:space-y-6">
          <Input
            label={t('search.productName')}
            placeholder={t('search.placeholder')}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={disabled || searching || !catalogUploaded}
            helperText={t('search.helper')}
            aria-required="true"
            aria-describedby="product-name-help"
          />

          <div className="space-y-2">
            <label
              htmlFor="search-method"
              className="block text-sm font-medium text-gray-700"
            >
              {t('search.method')}
            </label>
            <select
              id="search-method"
              value={searchMethod}
              onChange={(e) => setSearchMethod(e.target.value)}
              disabled={disabled || searching || !catalogUploaded}
              className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              aria-describedby="search-method-help"
            >
              <option value="hybrid">{t('search.method.hybrid')}</option>
              <option value="semantic">{t('search.method.semantic')}</option>
              <option value="fuzzy">{t('search.method.fuzzy')}</option>
              <option value="token">{t('search.method.token')}</option>
              <option value="hybrid_dbscan">{t('search.method.hybrid_dbscan')}</option>
              <option value="optimized_dbscan">{t('search.method.optimized_dbscan')}</option>
            </select>
            <p id="search-method-help" className="text-sm text-gray-500">
              {t('search.method.help')}
            </p>
          </div>
        </div>

        <div className="flex justify-center pt-2">
          <Button
            type="submit"
            variant="primary"
            size="lg"
            loading={searching}
            disabled={disabled || searching || !catalogUploaded || !query.trim()}
            className="w-full sm:w-auto sm:min-w-[200px]"
            aria-describedby={!catalogUploaded ? "catalog-required" : undefined}
          >
            <Search className="w-4 h-4 sm:w-5 sm:h-5 mr-2" />
            {searching ? t('search.button.loading') : t('search.button')}
          </Button>
        </div>
      </form>

      {searching && (
        <div className="text-center py-8">
          <LoadingSpinner size="lg" text={t('search.loading')} />
          <p className="text-sm text-gray-500 mt-2">
            {t('search.loadingHint')}
          </p>
        </div>
      )}
    </div>
  );
};

export default SearchInterface;
