import React, { useState } from 'react';
import { ChevronLeft, ChevronRight, Search } from 'lucide-react';
import { Button, Table } from './ui';
import { SearchResult } from '../types/api';
import { useI18n } from '../i18n';

interface ResultsDisplayProps {
  results: SearchResult[];
  query: string;
  onExport?: () => void;
  exporting?: boolean;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({
  results,
  query,
  onExport,
  exporting = false
}) => {
  const { t } = useI18n();
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(10);

  // Calculate pagination
  const totalPages = Math.ceil(results.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentResults = results.slice(startIndex, endIndex);

  // Format similarity score for display
  const formatSimilarityScore = (result: SearchResult): string => {
    const score = result.similarity_score || result.combined_score || 0;
    return `${(score * 100).toFixed(1)}%`;
  };

  // Determine search method used
  const getSearchMethod = (result: SearchResult): string => {
    if (result.search_method) return result.search_method;
    if (result.similarity_score) return t('search.method.badge.semantic');
    if (result.combined_score) return t('search.method.badge.hybrid');
    if (result.fuzzy_score) return t('search.method.badge.fuzzy');
    return t('search.method.badge.unknown');
  };

  // Get additional metrics for tooltip/details (currently unused but kept for future use)
  // const getAdditionalMetrics = (result: SearchResult): string => {
  //   const metrics: string[] = [];
  //
  //   if (result.cosine_score) {
  //     metrics.push(`Cosine: ${(result.cosine_score * 100).toFixed(1)}%`);
  //   }
  //   if (result.fuzzy_score) {
  //     metrics.push(`Fuzzy: ${result.fuzzy_score}`);
  //   }
  //   if (result.levenshtein_score) {
  //     metrics.push(`Levenshtein: ${result.levenshtein_score}`);
  //   }
  //
  //   return metrics.join(', ') || 'N/A';
  // };

  // Define table columns with responsive design
  const columns = [
    {
      key: 'rank',
      header: t('results.rank'),
      render: (value: any, row: SearchResult) => (
        <span className="font-medium text-gray-900 text-sm sm:text-base">
          #{row.rank || (startIndex + currentResults.indexOf(row) + 1)}
        </span>
      ),
      className: 'w-12 sm:w-16'
    },
    {
      key: 'document',
      header: t('results.foundAnalog'),
      render: (value: string) => (
        <div className="min-w-0 flex-1">
          <p className="text-xs sm:text-sm font-medium text-gray-900 break-words" title={value}>
            {value}
          </p>
        </div>
      )
    },
    {
      key: 'similarity_score',
      header: t('results.similarity'),
      render: (value: any, row: SearchResult) => {
        const score = row.similarity_score || row.combined_score || 0;
        const percentage = score * 100;

        let colorClass = 'text-red-600';
        if (percentage >= 80) colorClass = 'text-green-600';
        else if (percentage >= 60) colorClass = 'text-yellow-600';

        return (
          <div className="flex flex-col sm:flex-row sm:items-center space-y-1 sm:space-y-0 sm:space-x-2">
            <span className={`font-medium text-xs sm:text-sm ${colorClass}`}>
              {formatSimilarityScore(row)}
            </span>
            <div className="w-12 sm:w-16 bg-gray-200 rounded-full h-1.5 sm:h-2">
              <div
                className={`h-1.5 sm:h-2 rounded-full ${
                  percentage >= 80 ? 'bg-green-500' :
                  percentage >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                }`}
                style={{ width: `${Math.min(percentage, 100)}%` }}
                role="progressbar"
                aria-valuenow={percentage}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label={`${t('results.similarity')}: ${percentage.toFixed(1)}%`}
              />
            </div>
          </div>
        );
      },
      className: 'w-20 sm:w-32'
    },
    {
      key: 'search_method',
      header: t('results.method'),
      render: (value: any, row: SearchResult) => (
        <span className="inline-flex items-center px-1.5 py-0.5 sm:px-2.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
          <span className="hidden sm:inline">{getSearchMethod(row)}</span>
          <span className="sm:hidden">{getSearchMethod(row).charAt(0)}</span>
        </span>
      ),
      className: 'w-16 sm:w-24'
    },
    {
      key: 'document_id',
      header: t('results.id'),
      render: (value: string | number) => (
        <span className="text-xs sm:text-sm text-gray-500 font-mono break-all">
          {value}
        </span>
      ),
      className: 'w-16 sm:w-20 hidden sm:table-cell'
    }
  ];

  // Pagination controls
  const PaginationControls = () => {
    if (totalPages <= 1) return null;

    return (
      <div className="flex items-center justify-between px-4 py-3 bg-white border-t border-gray-200 sm:px-6">
        <div className="flex justify-between flex-1 sm:hidden">
          <Button
            variant="outline"
            onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
            disabled={currentPage === 1}
          >
            {t('common.previous')}
          </Button>
          <Button
            variant="outline"
            onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
            disabled={currentPage === totalPages}
          >
            {t('common.next')}
          </Button>
        </div>
        <div className="hidden sm:flex sm:flex-1 sm:items-center sm:justify-between">
          <div>
            <p className="text-sm text-gray-700">
              {t('results.showing', { from: startIndex + 1, to: Math.min(endIndex, results.length), total: results.length })}
            </p>
          </div>
          <div>
            <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="rounded-r-none"
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              
              {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                let pageNum: number;
                if (totalPages <= 5) {
                  pageNum = i + 1;
                } else if (currentPage <= 3) {
                  pageNum = i + 1;
                } else if (currentPage >= totalPages - 2) {
                  pageNum = totalPages - 4 + i;
                } else {
                  pageNum = currentPage - 2 + i;
                }
                
                return (
                  <Button
                    key={pageNum}
                    variant={currentPage === pageNum ? "primary" : "outline"}
                    size="sm"
                    onClick={() => setCurrentPage(pageNum)}
                    className="rounded-none"
                  >
                    {pageNum}
                  </Button>
                );
              })}
              
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                className="rounded-l-none"
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </nav>
          </div>
        </div>
      </div>
    );
  };

  if (results.length === 0) {
    return (
      <div className="text-center py-12">
        <Search className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">{t('search.noResults.title')}</h3>
        <p className="text-gray-500">
          {t('search.noResults.text', { query })}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">
            {t('search.results.title', { query })}
          </h2>
          <p className="text-sm text-gray-600">
            {t('search.results.count', { count: results.length, plural: results.length !== 1 ? 's' : '' })}
          </p>
        </div>
        
        {onExport && (
          <Button
            variant="secondary"
            onClick={onExport}
            loading={exporting}
            disabled={exporting}
          >
            {t('search.results.export')}
          </Button>
        )}
      </div>

      <div className="bg-white shadow overflow-hidden sm:rounded-md">
        <Table
          columns={columns}
          data={currentResults}
          emptyMessage={t('search.table.empty')}
        />
        <PaginationControls />
      </div>
    </div>
  );
};

export default ResultsDisplay;
