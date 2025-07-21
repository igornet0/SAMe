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

export interface UploadResponse {
  status: string;
  message: string;
  statistics?: Record<string, any>;
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
}
