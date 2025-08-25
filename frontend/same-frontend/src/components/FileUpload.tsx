import React, { useCallback, useState, useRef, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, File, CheckCircle, X, Square } from 'lucide-react';
import { Button, LoadingSpinner, Alert } from './ui';
import { FileUploadInfo, UploadResponse } from '../types/api';
import { useI18n } from '../i18n';

interface FileUploadProps {
  onUploadSuccess: (fileInfo: FileUploadInfo, response: UploadResponse) => void;
  onUploadError: (error: string) => void;
  disabled?: boolean;
}

type UploadMode = 'no_processing' | 'background_processing' | 'advanced_processing';

const FileUpload: React.FC<FileUploadProps> = ({
  onUploadSuccess,
  onUploadError,
  disabled = false
}) => {
  const { t } = useI18n();
  const [uploading, setUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<FileUploadInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pollingProgress, setPollingProgress] = useState<number>(0);
  const [mode, setMode] = useState<UploadMode>('background_processing');
  const [maxRows, setMaxRows] = useState<number | ''>('');
  const currentTaskIdRef = useRef<string | null>(null);
  const cancelRequestedRef = useRef<boolean>(false);

  // LocalStorage keys for resuming processing after reload
  const LS_KEYS = {
    taskId: 'same.upload.taskId',
    mode: 'same.upload.mode',
    fileName: 'same.upload.fileName',
    fileSize: 'same.upload.fileSize',
    fileType: 'same.upload.fileType',
    progress: 'same.upload.progress',
  } as const;

  const persistInProgress = (taskId: string, file: File, modeVal: UploadMode) => {
    try {
      localStorage.setItem(LS_KEYS.taskId, taskId);
      localStorage.setItem(LS_KEYS.mode, modeVal);
      localStorage.setItem(LS_KEYS.fileName, file.name);
      localStorage.setItem(LS_KEYS.fileSize, String(file.size));
      localStorage.setItem(LS_KEYS.fileType, file.type);
      localStorage.setItem(LS_KEYS.progress, String(pollingProgress));
    } catch {}
  };

  const updatePersistedProgress = (progress: number) => {
    try { localStorage.setItem(LS_KEYS.progress, String(progress)); } catch {}
  };

  const clearPersisted = () => {
    try {
      localStorage.removeItem(LS_KEYS.taskId);
      localStorage.removeItem(LS_KEYS.mode);
      localStorage.removeItem(LS_KEYS.fileName);
      localStorage.removeItem(LS_KEYS.fileSize);
      localStorage.removeItem(LS_KEYS.fileType);
      localStorage.removeItem(LS_KEYS.progress);
    } catch {}
  };

  const validateFile = (file: File): string | null => {
    const allowedTypes = [
      'text/csv',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/json'
    ];
    
    const allowedExtensions = ['.csv', '.xlsx', '.xls', '.json'];
    const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
    
    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
      return t('upload.validation.type');
    }
    
    if (file.size > 50 * 1024 * 1024) { // 50MB limit
      return t('upload.validation.size');
    }
    
    return null;
  };

  const pollUploadStatus = async (taskId: string): Promise<UploadResponse> => {
    const maxAttempts = 7200; // ~2 hours at 1s interval (увеличено для больших файлов)
    const intervalMs = 1000;
    let attempt = 0;
    currentTaskIdRef.current = taskId;
    cancelRequestedRef.current = false;
    while (attempt < maxAttempts) {
      if (cancelRequestedRef.current) {
        throw new Error('Processing canceled by user');
      }
      const res = await fetch(`http://localhost:8000/search/upload-status/${encodeURIComponent(taskId)}`);
      if (!res.ok) {
        const err = await res.text();
        throw new Error(err || 'Failed to check upload status');
      }
      const json: UploadResponse = await res.json();
      if (json.successful || json.status === 'success' || json.status === 'success_with_errors') {
        setPollingProgress(100);
        return json;
      }
      if (json.error || json.status === 'error') {
        throw new Error(json.error || json.message || 'Upload processing failed');
      }
      attempt += 1;
      // Обновляем прогресс каждые 10 секунд
      if (attempt % 10 === 0) {
        const progress = Math.min((attempt / maxAttempts) * 100, 95); // Максимум 95% пока не завершится
        setPollingProgress(progress);
        updatePersistedProgress(progress);
      }
      await new Promise(r => setTimeout(r, intervalMs));
    }
    throw new Error('Processing timed out');
  };

  const requestCancel = async () => {
    try {
      cancelRequestedRef.current = true;
      const taskId = currentTaskIdRef.current;
      if (taskId) {
        await fetch(`http://localhost:8000/search/cancel-upload/${encodeURIComponent(taskId)}`, { method: 'POST' });
      }
    } catch (e) {
      // ignore
    } finally {
      setUploading(false);
      clearPersisted();
    }
  };

  const uploadFile = async (file: File) => {
    setUploading(true);
    setError(null);
    setPollingProgress(0);
    currentTaskIdRef.current = null;
    cancelRequestedRef.current = false;
    
    try {
      const formData = new FormData();
      formData.append('file', file);

      let final: UploadResponse;
      if (mode === 'no_processing') {
        // Отправляем на бэк без фоновой обработки (как "принять файл" — тут можно расширить на простой парсинг и показ статистики)
        const res = await fetch('http://localhost:8000/search/upload-catalog', { method: 'POST', body: formData });
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || 'Upload failed');
        }
        const json: UploadResponse = await res.json();
        if (json.task_id) {
          currentTaskIdRef.current = json.task_id;
          persistInProgress(json.task_id, file, mode);
          final = await pollUploadStatus(json.task_id);
        } else {
          final = json;
        }
      } else if (mode === 'background_processing') {
        // Существующий режим: фоновая задача через Celery
        const res = await fetch('http://localhost:8000/search/upload-catalog', { method: 'POST', body: formData });
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || 'Upload failed');
        }
        const json: UploadResponse = await res.json();
        if (json.task_id) {
          currentTaskIdRef.current = json.task_id;
          persistInProgress(json.task_id, file, mode);
          final = await pollUploadStatus(json.task_id);
        } else {
          final = json;
        }
      } else {
        // advanced_processing: синхронная обработка через новый эндпоинт
        const url = new URL('http://localhost:8000/search/process-catalog-advanced');
        if (maxRows !== '' && Number(maxRows) > 0) url.searchParams.set('max_rows', String(maxRows));
        const res = await fetch(url.toString(), { method: 'POST', body: formData });
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || 'Advanced processing failed');
        }
        final = await res.json();
      }

      const dataset = (final.statistics as any)?.dataset;
      const engine = (final.statistics as any)?.engine;
      const recordCount = dataset?.total_rows ?? engine?.catalog_size;
      const fileInfo: FileUploadInfo = {
        name: file.name,
        size: file.size,
        type: file.type,
        recordCount: recordCount ?? undefined
      };

      setUploadedFile(fileInfo);
      onUploadSuccess(fileInfo, final);
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Upload failed';
      setError(errorMessage);
      onUploadError(errorMessage);
    } finally {
      setUploading(false);
      clearPersisted();
    }
  };

  // Resume processing after page reload if a task is in progress
  useEffect(() => {
    try {
      const savedTaskId = localStorage.getItem(LS_KEYS.taskId);
      if (savedTaskId) {
        const savedMode = (localStorage.getItem(LS_KEYS.mode) as UploadMode) || 'background_processing';
        const savedName = localStorage.getItem(LS_KEYS.fileName) || 'catalog';
        const savedSize = Number(localStorage.getItem(LS_KEYS.fileSize) || '0');
        const savedType = localStorage.getItem(LS_KEYS.fileType) || '';
        const savedProgress = Number(localStorage.getItem(LS_KEYS.progress) || '0');

        setMode(savedMode);
        setUploading(true);
        setPollingProgress(Number.isFinite(savedProgress) ? savedProgress : 0);
        currentTaskIdRef.current = savedTaskId;

        // Resume polling
        (async () => {
          try {
            const resumed = await pollUploadStatus(savedTaskId);
            const dataset = (resumed.statistics as any)?.dataset;
            const engine = (resumed.statistics as any)?.engine;
            const recordCount = dataset?.total_rows ?? engine?.catalog_size;
            const fileInfo: FileUploadInfo = {
              name: savedName,
              size: savedSize,
              type: savedType,
              recordCount: recordCount ?? undefined,
            };
            setUploadedFile(fileInfo);
            onUploadSuccess(fileInfo, resumed);
          } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Upload failed';
            setError(errorMessage);
            onUploadError(errorMessage);
          } finally {
            setUploading(false);
            clearPersisted();
          }
        })();
      }
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    const validationError = validateFile(file);

    if (validationError) {
      setError(validationError);
      onUploadError(validationError);
      return;
    }

    uploadFile(file);
  }, [onUploadSuccess, onUploadError, uploadFile]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/json': ['.json']
    },
    multiple: false,
    disabled: disabled || uploading
  });

  const clearFile = () => {
    setUploadedFile(null);
    setError(null);
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-4">
      {/* Режим загрузки */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Режим загрузки</label>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as UploadMode)}
            className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            disabled={disabled || uploading}
          >
            <option value="no_processing">Без обработки (только загрузка/статистика)</option>
            <option value="background_processing">Фоновая обработка (рекомендуется)</option>
            <option value="advanced_processing">Синхронная расширенная обработка</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Лимит строк (для синхронной обработки)</label>
          <input
            type="number"
            min={0}
            placeholder="например, 2000"
            value={maxRows}
            onChange={(e) => setMaxRows(e.target.value === '' ? '' : Number(e.target.value))}
            className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            disabled={disabled || uploading || mode !== 'advanced_processing'}
          />
        </div>
        <div className="flex items-end">
          <Button
            variant="outline"
            size="sm"
            onClick={requestCancel}
            disabled={!uploading}
            className="text-red-600 border-red-300 hover:text-red-700 hover:border-red-400"
          >
            <Square className="h-4 w-4 mr-1" /> Отменить обработку
          </Button>
        </div>
      </div>
      <div className="text-center">
        <h2 className="text-lg font-semibold text-gray-900 mb-2">
          {t('upload.title')}
        </h2>
        <p className="text-sm text-gray-600 mb-4">
          {t('upload.subtitle')}
        </p>
      </div>

      {error && (
        <Alert
          type="error"
          message={error}
          onClose={() => setError(null)}
        />
      )}

      {!uploadedFile ? (
        <div
          {...getRootProps()}
          className={`
            border-2 border-dashed rounded-lg p-4 sm:p-6 lg:p-8 text-center cursor-pointer transition-colors
            ${isDragActive
              ? 'border-blue-400 bg-blue-50'
              : 'border-gray-300 hover:border-gray-400'
            }
            ${disabled || uploading ? 'opacity-50 cursor-not-allowed' : ''}
          `}
          role="button"
          tabIndex={0}
          aria-label={t('upload.aria')}
          aria-describedby="upload-description"
        >
          <input
            {...getInputProps()}
            aria-label={t('upload.inputAria')}
          />

          {uploading ? (
            <div role="status" aria-live="polite">
              <LoadingSpinner size="lg" text={t('upload.uploading')} />
              {pollingProgress > 0 && (
                <div className="mt-4">
                  <div className="flex justify-between text-sm text-gray-600 mb-1">
                    <span>{t('upload.processing')}</span>
                    <span>{Math.round(pollingProgress)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300" 
                      style={{ width: `${pollingProgress}%` }}
                    />
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    {t('upload.processingHint')}
                  </p>
                </div>
              )}
            </div>
          ) : (
            <>
              <Upload className="mx-auto h-8 w-8 sm:h-10 sm:w-10 lg:h-12 lg:w-12 text-gray-400 mb-2 sm:mb-4" />
              <div className="space-y-1 sm:space-y-2">
                <p className="text-base sm:text-lg font-medium text-gray-900">
                  {isDragActive ? t('upload.dropNow') : t('upload.dropHere')}
                </p>
                <p id="upload-description" className="text-xs sm:text-sm text-gray-500 px-2">
                  {t('upload.supports')}
                </p>
              </div>
            </>
          )}
        </div>
      ) : (
        <div className="border border-green-200 bg-green-50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <CheckCircle className="h-6 w-6 text-green-600" />
              <div>
                <p className="font-medium text-green-900">{uploadedFile.name}</p>
                <p className="text-sm text-green-700">
                  {formatFileSize(uploadedFile.size)}
                  {uploadedFile.recordCount && ` • ${uploadedFile.recordCount} ${t('common.records')}`}
                </p>
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={clearFile}
              className="text-gray-600 hover:text-gray-800"
            >
              <X className="h-4 w-4 mr-1" />
              {t('common.remove')}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;
