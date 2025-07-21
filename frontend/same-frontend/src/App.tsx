import React, { useState, useEffect, useCallback } from 'react';
import { CheckCircle } from 'lucide-react';
import FileUpload from './components/FileUpload';
import SearchInterface from './components/SearchInterface';
import ResultsDisplay from './components/ResultsDisplay';
import ErrorBoundary from './components/ErrorBoundary';
import { Alert, LoadingSpinner, ToastProvider, useToast } from './components/ui';
import { FileUploadInfo, UploadResponse, SearchResult, AppState } from './types/api';
import { downloadExcelFile } from './utils/exportUtils';
import apiClient from './services/apiClient';

const AppContent: React.FC = () => {
  const [appState, setAppState] = useState<AppState>({
    catalogUploaded: false,
    isSearching: false,
    isExporting: false,
    uploadedFile: null,
    searchResults: [],
    currentQuery: '',
    error: null
  });

  const [connectionStatus, setConnectionStatus] = useState<'checking' | 'connected' | 'error'>('checking');
  const { showSuccess, showError, showWarning, showInfo } = useToast();

  const checkBackendConnection = useCallback(async () => {
    try {
      await apiClient.healthCheck();
      setConnectionStatus('connected');
      showInfo('Connected to SAMe backend successfully');
    } catch (error) {
      console.error('Backend connection failed:', error);
      setConnectionStatus('error');
      showError('Failed to connect to SAMe backend. Please ensure the server is running.');
    }
  }, [showError, showInfo]);

  // Check backend connection on mount
  useEffect(() => {
    checkBackendConnection();
  }, [checkBackendConnection]);

  const handleUploadSuccess = (fileInfo: FileUploadInfo, response: UploadResponse) => {
    setAppState(prev => ({
      ...prev,
      catalogUploaded: true,
      uploadedFile: fileInfo,
      error: null
    }));
    showSuccess(
      `Catalog uploaded successfully! ${fileInfo.recordCount ? `${fileInfo.recordCount} records loaded.` : ''}`,
      'Upload Complete'
    );
  };

  const handleUploadError = (error: string) => {
    setAppState(prev => ({
      ...prev,
      catalogUploaded: false,
      uploadedFile: null,
      error
    }));
    showError(error, 'Upload Failed');
  };

  const handleSearchResults = (results: SearchResult[], query: string) => {
    setAppState(prev => ({
      ...prev,
      searchResults: results,
      currentQuery: query,
      error: null
    }));

    if (results.length === 0) {
      showWarning(`No analogs found for "${query}". Try adjusting your search terms.`, 'No Results');
    } else {
      showSuccess(`Found ${results.length} analog${results.length !== 1 ? 's' : ''} for "${query}"`, 'Search Complete');
    }
  };

  const handleSearchError = (error: string) => {
    setAppState(prev => ({
      ...prev,
      error
    }));
    showError(error, 'Search Failed');
  };

  const handleExport = async () => {
    if (appState.searchResults.length === 0) return;

    setAppState(prev => ({ ...prev, isExporting: true, error: null }));

    try {
      await downloadExcelFile(
        appState.searchResults,
        appState.currentQuery,
        `analog_search_${appState.currentQuery.replace(/[^a-zA-Z0-9]/g, '_')}_${new Date().toISOString().split('T')[0]}.xlsx`
      );
      showSuccess('Excel file downloaded successfully!', 'Export Complete');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Export failed';
      setAppState(prev => ({ ...prev, error: errorMessage }));
      showError(errorMessage, 'Export Failed');
    } finally {
      setAppState(prev => ({ ...prev, isExporting: false }));
    }
  };

  const clearError = () => {
    setAppState(prev => ({ ...prev, error: null }));
  };

  if (connectionStatus === 'checking') {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <LoadingSpinner size="lg" text="Connecting to SAMe backend..." />
      </div>
    );
  }

  if (connectionStatus === 'error') {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-md w-full">
          <Alert
            type="error"
            title="Backend Connection Failed"
            message="Unable to connect to the SAMe backend server. Please ensure the server is running on http://localhost:8000"
          />
          <div className="mt-4 text-center">
            <button
              onClick={checkBackendConnection}
              className="text-blue-600 hover:text-blue-800 font-medium"
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200" role="banner">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-2 sm:space-y-0">
            <div>
              <h1 className="text-xl sm:text-2xl font-bold text-gray-900">
                SAMe - Analog Search Engine
              </h1>
              <p className="text-sm text-gray-600 mt-1">
                Search for material and technical resource analogs
              </p>
            </div>
            <div className="flex items-center space-x-2" role="status" aria-live="polite">
              <CheckCircle className="h-4 w-4 sm:h-5 sm:w-5 text-green-500" aria-hidden="true" />
              <span className="text-sm text-gray-600">Connected</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-6 lg:py-8" role="main">
        <div className="space-y-4 sm:space-y-6 lg:space-y-8">
          {/* Global Error Alert */}
          {appState.error && (
            <div role="alert">
              <Alert
                type="error"
                message={appState.error}
                onClose={clearError}
              />
            </div>
          )}

          {/* File Upload Section */}
          <section className="bg-white rounded-lg shadow-sm p-4 sm:p-6" aria-labelledby="upload-section">
            <FileUpload
              onUploadSuccess={handleUploadSuccess}
              onUploadError={handleUploadError}
              disabled={appState.isSearching || appState.isExporting}
            />
          </section>

          {/* Search Section */}
          <section className="bg-white rounded-lg shadow-sm p-4 sm:p-6" aria-labelledby="search-section">
            <SearchInterface
              catalogUploaded={appState.catalogUploaded}
              onSearchResults={handleSearchResults}
              onSearchError={handleSearchError}
              disabled={appState.isExporting}
            />
          </section>

          {/* Results Section */}
          {appState.searchResults.length > 0 && (
            <section className="bg-white rounded-lg shadow-sm p-4 sm:p-6" aria-labelledby="results-section">
              <ResultsDisplay
                results={appState.searchResults}
                query={appState.currentQuery}
                onExport={handleExport}
                exporting={appState.isExporting}
              />
            </section>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <p className="text-center text-sm text-gray-500">
            SAMe v1.2.0 - Search Analog Model Engine
          </p>
        </div>
      </footer>
    </div>
  );
};

function App() {
  return (
    <ErrorBoundary>
      <ToastProvider>
        <AppContent />
      </ToastProvider>
    </ErrorBoundary>
  );
}

export default App;
