import React, { useMemo, useState } from 'react';
import { DatasetStatistics, DatasetTopValue } from '../types/api';
import { useI18n } from '../i18n';

interface DatasetStatsProps {
  dataset: DatasetStatistics | undefined;
  processingReport?: { failed_count: number; failed_rows: { code: string | number; name: string; error?: string }[] } | null;
}

const DatasetStats: React.FC<DatasetStatsProps> = ({ dataset, processingReport }) => {
  const [selectedColumn, setSelectedColumn] = useState<string | ''>('');
  const [selectedValue, setSelectedValue] = useState<string | number | boolean | null | ''>('');
  const { t } = useI18n();

  const columnStats = useMemo(() => {
    if (!dataset || !selectedColumn) return undefined;
    return dataset.per_column[selectedColumn];
  }, [dataset, selectedColumn]);

  const totalRows = dataset?.total_rows ?? 0;

  const selectedValueInfo = useMemo(() => {
    if (!columnStats || selectedValue === '') return undefined;
    const tv = columnStats.top_values.find((t: DatasetTopValue) => {
      // Compare with loose equality after stringifying for robustness
      const normalize = (v: any) => (v === null || v === undefined ? '' : String(v));
      return normalize(t.value) === normalize(selectedValue);
    });
    return tv;
  }, [columnStats, selectedValue]);

  if (!dataset) {
    return null;
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">{t('dataset.title')}</h2>
          <p className="text-sm text-gray-600">{t('dataset.totalRows', { total: totalRows, cols: dataset.columns.length })}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Columns list */}
        <div className="border rounded-lg p-3">
          <h3 className="font-medium mb-2">{t('dataset.columns')}</h3>
          <ul className="max-h-64 overflow-auto divide-y">
            {dataset.columns.map((col) => {
              const stats = dataset.per_column[col];
              return (
                <li key={col} className="py-2 flex items-center justify-between">
                  <button
                    type="button"
                    onClick={() => { setSelectedColumn(col); setSelectedValue(''); }}
                    className={`text-left truncate pr-2 ${selectedColumn === col ? 'font-semibold text-blue-600' : 'text-gray-800'}`}
                    title={col}
                  >
                    {col}
                  </button>
                  <span className="text-xs text-gray-500">{t('dataset.missing', { count: stats.missing_count, pct: stats.missing_pct.toFixed(1) })}</span>
                </li>
              );
            })}
          </ul>
        </div>

        {/* Selected column details */}
        <div className="border rounded-lg p-3 md:col-span-2">
          <h3 className="font-medium mb-2">{t('dataset.columnDetails')}</h3>
          {selectedColumn ? (
            <div className="space-y-3">
              <div className="text-sm text-gray-700">
                <span className="font-semibold">{selectedColumn}</span>
                {columnStats && (
                  <>
                    <span className="ml-2">{t('dataset.dtype')}: {columnStats.dtype}</span>
                    <span className="ml-2">{t('dataset.unique')}: {columnStats.unique_count}</span>
                    <span className="ml-2">{t('dataset.missingShort')}: {columnStats.missing_count} ({columnStats.missing_pct.toFixed(1)}%)</span>
                  </>
                )}
              </div>

              {/* Filter by value */}
              {columnStats && (
                <div className="space-y-2">
                  <label className="block text-sm text-gray-700">{t('dataset.filterLabel')}</label>
                  <select
                    className="w-full border rounded px-2 py-1"
                    value={selectedValue === '' ? '' : String(selectedValue)}
                    onChange={(e) => setSelectedValue(e.target.value)}
                  >
                    <option value="">{t('dataset.selectValue')}</option>
                    {columnStats.top_values.map((tv, idx) => (
                      <option key={`${idx}-${String(tv.value)}`} value={String(tv.value)}>
                        {tv.value === null ? t('common.null') : String(tv.value)} ({tv.count})
                      </option>
                    ))}
                  </select>

                  {selectedValueInfo && (
                    <div className="text-sm text-gray-700">
                      {t('dataset.rowsWithValue', { value: selectedValue === '' ? '' : String(selectedValue) })}
                      <span className="ml-1 font-semibold">{selectedValueInfo.count}</span>
                      <span className="ml-1 text-gray-500">({((selectedValueInfo.count / Math.max(1, totalRows)) * 100).toFixed(1)}%)</span>
                    </div>
                  )}
                </div>
              )}

              {/* Top values table */}
              {columnStats && (
                <div className="overflow-auto border rounded">
                  <table className="min-w-full text-sm">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="text-left px-2 py-1">{t('common.value')}</th>
                        <th className="text-left px-2 py-1">{t('common.count')}</th>
                        <th className="text-left px-2 py-1">{t('common.percent')}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {columnStats.top_values.map((tv, idx) => (
                        <tr key={idx} className="border-t">
                          <td className="px-2 py-1 truncate max-w-xs" title={tv.value === null ? t('common.null') : String(tv.value)}>
                            {tv.value === null ? t('common.null') : String(tv.value)}
                          </td>
                          <td className="px-2 py-1">{tv.count}</td>
                          <td className="px-2 py-1">{((tv.count / Math.max(1, totalRows)) * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-gray-600">{t('dataset.columnDetails')}</p>
          )}
        </div>
      </div>

      {processingReport && (
        <div className="border rounded-lg p-3">
          <h3 className="font-medium mb-2">{t('dataset.processingReport')}</h3>
          <p className="text-sm text-gray-600">{t('dataset.failedRows', { count: processingReport.failed_count })}</p>
          {processingReport.failed_count > 0 && (
            <div className="mb-2 text-sm text-yellow-800 bg-yellow-50 border border-yellow-200 rounded p-2">
              {t('dataset.failedHint')}
            </div>
          )}
          {processingReport.failed_rows && processingReport.failed_rows.length > 0 && (
            <details className="mt-2">
              <summary className="cursor-pointer text-sm text-blue-600">{t('dataset.showFailed', { count: processingReport.failed_rows.length })}</summary>
              <div className="mt-2 max-h-64 overflow-auto border rounded">
                <table className="min-w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="text-left px-2 py-1">{t('common.code')}</th>
                      <th className="text-left px-2 py-1">{t('common.name')}</th>
                      <th className="text-left px-2 py-1">{t('common.error')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {processingReport.failed_rows.map((r, idx) => (
                      <tr key={idx} className="border-t">
                        <td className="px-2 py-1 whitespace-nowrap">{String(r.code)}</td>
                        <td className="px-2 py-1">{r.name}</td>
                        <td className="px-2 py-1 text-gray-600">{r.error || '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
};

export default DatasetStats;


