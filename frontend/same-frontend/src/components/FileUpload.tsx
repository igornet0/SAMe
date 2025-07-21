import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, File, CheckCircle, X } from 'lucide-react';
import { Button, LoadingSpinner, Alert } from './ui';
import { FileUploadInfo, UploadResponse } from '../types/api';

interface FileUploadProps {
  onUploadSuccess: (fileInfo: FileUploadInfo, response: UploadResponse) => void;
  onUploadError: (error: string) => void;
  disabled?: boolean;
}

const FileUpload: React.FC<FileUploadProps> = ({
  onUploadSuccess,
  onUploadError,
  disabled = false
}) => {
  const [uploading, setUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<FileUploadInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

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
      return 'Only CSV, Excel (.xlsx, .xls), and JSON files are supported';
    }
    
    if (file.size > 50 * 1024 * 1024) { // 50MB limit
      return 'File size must be less than 50MB';
    }
    
    return null;
  };

  const uploadFile = async (file: File) => {
    setUploading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('http://localhost:8000/search/upload-catalog', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }
      
      const result: UploadResponse = await response.json();
      
      const fileInfo: FileUploadInfo = {
        name: file.name,
        size: file.size,
        type: file.type,
        recordCount: result.statistics?.total_items || undefined
      };
      
      setUploadedFile(fileInfo);
      onUploadSuccess(fileInfo, result);
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Upload failed';
      setError(errorMessage);
      onUploadError(errorMessage);
    } finally {
      setUploading(false);
    }
  };

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
      <div className="text-center">
        <h2 className="text-lg font-semibold text-gray-900 mb-2">
          Upload Catalog File
        </h2>
        <p className="text-sm text-gray-600 mb-4">
          Upload your catalog file in CSV, Excel, or JSON format
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
          aria-label="Upload catalog file"
          aria-describedby="upload-description"
        >
          <input
            {...getInputProps()}
            aria-label="File upload input"
          />

          {uploading ? (
            <div role="status" aria-live="polite">
              <LoadingSpinner size="lg" text="Uploading..." />
            </div>
          ) : (
            <>
              <Upload className="mx-auto h-8 w-8 sm:h-10 sm:w-10 lg:h-12 lg:w-12 text-gray-400 mb-2 sm:mb-4" />
              <div className="space-y-1 sm:space-y-2">
                <p className="text-base sm:text-lg font-medium text-gray-900">
                  {isDragActive ? 'Drop the file here' : 'Drop your file here, or click to browse'}
                </p>
                <p id="upload-description" className="text-xs sm:text-sm text-gray-500 px-2">
                  Supports CSV, Excel (.xlsx, .xls), and JSON files up to 50MB
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
                  {uploadedFile.recordCount && ` • ${uploadedFile.recordCount} records`}
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
              Remove
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;
