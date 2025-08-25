// API Response Types for SAMe Analog Search

export interface SearchResult {
  document_id: string | number;
  document: string;
  similarity_score?: number;
  combined_score?: number;
  fuzzy_score?: number;
  cosine_score?: number;
  levenshtein_score?: number;
  search_method?: string;
  rank?: number;
  extracted_parameters?: Record<string, any>;
}

export interface SearchResponse {
  results: Record<string, SearchResult[]>;
  statistics: Record<string, any>;
  processing_time: number;
}

export interface SearchRequest {
  queries: string[];
  method?: string;
  similarity_threshold?: number;
  max_results?: number;
}

export interface InitializeRequest {
  catalog_file_path?: string;
  search_method?: string;
  similarity_threshold?: number;
}

export interface DatasetTopValue {
  value: string | number | boolean | null;
  count: number;
}

export interface DatasetPerColumnStats {
  dtype: string;
  non_null_count: number;
  missing_count: number;
  missing_pct: number;
  unique_count: number;
  top_values: DatasetTopValue[];
}

export interface DatasetStatistics {
  total_rows: number;
  columns: string[];
  per_column: Record<string, DatasetPerColumnStats>;
}

export interface EngineStatistics {
  is_ready: boolean;
  catalog_size: number;
  search_method: string;
  similarity_threshold: number;
  [key: string]: any;
}

export interface UploadResponse {
  status: string; // queued | success | error | ...
  message?: string;
  task_id?: string;
  state?: string;
  ready?: boolean;
  successful?: boolean;
  error?: string;
  statistics?: {
    engine?: EngineStatistics;
    dataset?: DatasetStatistics;
    processing_report?: {
      failed_count: number;
      failed_rows: { code: string | number; name: string; error?: string }[];
    } | null;
    [key: string]: any;
  } | Record<string, any>;
  // Поля для расширенной синхронной обработки
  output_csv_path?: string;
  columns?: string[];
  rows_preview?: Record<string, any>[];
  preview_limit?: number;
}

export interface ExportResult {
  Raw_Name: string;
  Cleaned_Name: string;
  Lemmatized_Name: string;
  Normalized_Name: string;
  Candidate_Name: string;
  Similarity_Score: number | string;
  Relation_Type: string;
  Suggested_Category: string;
  Final_Decision: string;
  Comment: string;
}

export interface ApiError {
  detail: string;
  status_code?: number;
}

// File upload types
export interface FileUploadInfo {
  name: string;
  size: number;
  type: string;
  recordCount?: number;
}

// Application state types
export interface AppState {
  catalogUploaded: boolean;
  isSearching: boolean;
  isExporting: boolean;
  uploadedFile: FileUploadInfo | null;
  searchResults: SearchResult[];
  currentQuery: string;
  error: string | null;
  uploadStatistics?: UploadResponse['statistics'];
}
