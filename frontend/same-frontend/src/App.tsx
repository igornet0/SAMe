import React, { useState, useEffect, useCallback } from 'react';
import { CheckCircle } from 'lucide-react';
import FileUpload from './components/FileUpload';
import SearchInterface from './components/SearchInterface';
import ResultsDisplay from './components/ResultsDisplay';
import ErrorBoundary from './components/ErrorBoundary';
import { Alert, LoadingSpinner, ToastProvider, useToast } from './components/ui';
import { useI18n, LanguageSwitcher } from './i18n';
import { FileUploadInfo, UploadResponse, SearchResult, AppState, DatasetStatistics } from './types/api';
import DatasetStats from './components/DatasetStats';
import { downloadExcelFile } from './utils/exportUtils';

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
  const { t } = useI18n();

  const checkBackendConnection = useCallback(async () => {
    try {
      // Try lightweight healthz first, then fallback to /search/
      let res = await fetch('/healthz');
      if (!res.ok) {
        res = await fetch('/search/health');
      }
      if (!res.ok) {
        res = await fetch('/search/');
      }
      if (!res.ok) throw new Error('Health check failed');
      setConnectionStatus('connected');
      if (!(typeof process !== 'undefined' && process.env && process.env.NODE_ENV === 'test')) {
        showInfo(t('app.connectedToast.message'), t('app.connectedToast.title'));
      }
    } catch (error) {
      console.error('Backend connection failed:', error);
      setConnectionStatus('error');
      showError(t('app.connectError.message'), t('app.connectError.title'));
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
      error: null,
      uploadStatistics: response.statistics
    }));
    const countText = fileInfo.recordCount ? t('upload.successToast.count', { count: fileInfo.recordCount }) : '';
    const hasErrors = (response.statistics as any)?.processing_report?.failed_count > 0
      || (response.status === 'success_with_errors');
    if (hasErrors) {
      showWarning(
        t('upload.partialSuccessToast.message', { count: countText }),
        t('upload.partialSuccessToast.title')
      );
    } else {
      showSuccess(
        t('upload.successToast.message', { count: countText }),
        t('upload.successToast.title')
      );
    }
  };

  const handleUploadError = (error: string) => {
    setAppState(prev => ({
      ...prev,
      catalogUploaded: false,
      uploadedFile: null,
      error
    }));
    showError(error, t('upload.errorToast.title'));
  };

  const handleSearchResults = (results: SearchResult[], query: string) => {
    setAppState(prev => ({
      ...prev,
      searchResults: results,
      currentQuery: query,
      error: null
    }));

    if (results.length === 0) {
      showWarning(t('search.noResults.text', { query }), t('search.noResults.title'));
    } else {
      const plural = results.length !== 1 ? 's' : '';
      showSuccess(t('search.results.count', { count: results.length, plural }), t('search.title'));
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
      showSuccess('Excel file downloaded successfully!', t('search.results.export'));
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
        <LoadingSpinner size="lg" text={t('app.connecting')} />
      </div>
    );
  }

  if (connectionStatus === 'error') {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-md w-full">
          <Alert
            type="error"
            title={t('app.connectError.title')}
            message={t('app.connectError.message')}
          />
          <div className="mt-4 text-center">
            <button
              onClick={checkBackendConnection}
              className="text-blue-600 hover:text-blue-800 font-medium"
            >
              {t('common.tryAgain')}
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
                {t('app.title')}
              </h1>
              <p className="text-sm text-gray-600 mt-1">
                {t('app.subtitle')}
              </p>
            </div>
            <div className="flex items-center space-x-4" role="status" aria-live="polite">
              <CheckCircle className="h-4 w-4 sm:h-5 sm:w-5 text-green-500" aria-hidden="true" />
              <span className="text-sm text-gray-600">{t('common.connected')}</span>
              <LanguageSwitcher />
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

          {/* Dataset Statistics Section */}
          {appState.uploadStatistics && (
            <section className="bg-white rounded-lg shadow-sm p-4 sm:p-6" aria-labelledby="dataset-stats-section">
              <DatasetStats 
                dataset={(appState.uploadStatistics as any)?.dataset as DatasetStatistics}
                processingReport={(appState.uploadStatistics as any)?.processing_report as any}
              />
            </section>
          )}

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
            {t('app.footer')}
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
