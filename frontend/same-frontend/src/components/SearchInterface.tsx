import React, { useState } from 'react';
import { Search } from 'lucide-react';
import { Button, Input, Alert, LoadingSpinner } from './ui';
import { SearchResult, SearchResponse, SearchRequest } from '../types/api';

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
  const [query, setQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchMethod, setSearchMethod] = useState<string>('hybrid');

  const validateQuery = (searchQuery: string): string | null => {
    if (!searchQuery.trim()) {
      return 'Please enter a product name to search for';
    }
    
    if (searchQuery.trim().length < 2) {
      return 'Search query must be at least 2 characters long';
    }
    
    return null;
  };

  const performSearch = async () => {
    if (!catalogUploaded) {
      const errorMsg = 'Please upload a catalog file before searching';
      setError(errorMsg);
      onSearchError(errorMsg);
      return;
    }

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
        const errorData = await response.json();
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
          message="Please upload a catalog file first before searching for analogs"
        />
      )}

      <form onSubmit={handleSubmit} className="space-y-4 sm:space-y-6">
        <div className="space-y-4 sm:space-y-6">
          <Input
            label="Product Name"
            placeholder="Enter product name (e.g., 'Болт М10х50')"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={disabled || searching || !catalogUploaded}
            helperText="Enter the name of the product you want to find analogs for"
            aria-required="true"
            aria-describedby="product-name-help"
          />

          <div className="space-y-2">
            <label
              htmlFor="search-method"
              className="block text-sm font-medium text-gray-700"
            >
              Search Method
            </label>
            <select
              id="search-method"
              value={searchMethod}
              onChange={(e) => setSearchMethod(e.target.value)}
              disabled={disabled || searching || !catalogUploaded}
              className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              aria-describedby="search-method-help"
            >
              <option value="hybrid">Hybrid (Recommended)</option>
              <option value="semantic">Semantic Search</option>
              <option value="fuzzy">Fuzzy Search</option>
            </select>
            <p id="search-method-help" className="text-sm text-gray-500">
              Hybrid search combines semantic and fuzzy matching for best results
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
            {searching ? 'Searching...' : 'Search Analogs'}
          </Button>
        </div>
      </form>

      {searching && (
        <div className="text-center py-8">
          <LoadingSpinner size="lg" text="Searching for analogs..." />
          <p className="text-sm text-gray-500 mt-2">
            This may take a few moments depending on catalog size
          </p>
        </div>
      )}
    </div>
  );
};

export default SearchInterface;
