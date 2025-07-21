import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';
import { 
  SearchRequest, 
  SearchResponse, 
  UploadResponse, 
  InitializeRequest,
  ApiError 
} from '../types/api';

class ApiClient {
  private client: AxiosInstance;
  private baseURL: string;
  private maxRetries: number = 3;
  private retryDelay: number = 1000; // 1 second

  constructor(baseURL: string = 'http://localhost:8000') {
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 30000, // 30 seconds
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
        return config;
      },
      (error) => {
        console.error('Request error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        console.log(`Response received from ${response.config.url}:`, response.status);
        return response;
      },
      (error: AxiosError) => {
        console.error('Response error:', error);
        return Promise.reject(this.handleError(error));
      }
    );
  }

  private handleError(error: AxiosError): ApiError {
    if (error.response) {
      // Server responded with error status
      const responseData = error.response.data as any;
      const apiError: ApiError = {
        detail: responseData?.detail || error.message,
        status_code: error.response.status
      };
      return apiError;
    } else if (error.request) {
      // Request was made but no response received
      return {
        detail: 'Network error: Unable to connect to server. Please check your connection.',
        status_code: 0
      };
    } else {
      // Something else happened
      return {
        detail: error.message || 'An unexpected error occurred',
        status_code: 0
      };
    }
  }

  private async retryRequest<T>(
    requestFn: () => Promise<T>,
    retries: number = this.maxRetries
  ): Promise<T> {
    try {
      return await requestFn();
    } catch (error) {
      if (retries > 0 && this.shouldRetry(error as ApiError)) {
        console.log(`Retrying request... ${this.maxRetries - retries + 1}/${this.maxRetries}`);
        await this.delay(this.retryDelay);
        return this.retryRequest(requestFn, retries - 1);
      }
      throw error;
    }
  }

  private shouldRetry(error: ApiError): boolean {
    // Retry on network errors or server errors (5xx)
    return error.status_code === 0 || (error.status_code !== undefined && error.status_code >= 500 && error.status_code < 600);
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Health check
  async healthCheck(): Promise<{ status: string; message: string }> {
    return this.retryRequest(async () => {
      const response = await this.client.get('/search/');
      return response.data;
    });
  }

  // Upload catalog file
  async uploadCatalog(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    return this.retryRequest(async () => {
      const response = await this.client.post('/search/upload-catalog', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // 1 minute for file upload
      });
      return response.data;
    });
  }

  // Initialize search engine
  async initializeSearchEngine(request: InitializeRequest): Promise<any> {
    return this.retryRequest(async () => {
      const response = await this.client.post('/search/initialize', request);
      return response.data;
    });
  }

  // Search for analogs
  async searchAnalogs(request: SearchRequest): Promise<SearchResponse> {
    return this.retryRequest(async () => {
      const response = await this.client.post('/search/search-analogs', request);
      return response.data;
    });
  }

  // Search single analog (alternative endpoint)
  async searchSingleAnalog(
    query: string, 
    method: string = 'hybrid', 
    maxResults: number = 10
  ): Promise<{ query: string; results: any[]; method: string }> {
    return this.retryRequest(async () => {
      const response = await this.client.get(`/search/search-single/${encodeURIComponent(query)}`, {
        params: { method, max_results: maxResults }
      });
      return response.data;
    });
  }

  // Export results
  async exportResults(
    results: Record<string, any[]>, 
    format: string = 'excel'
  ): Promise<Blob> {
    return this.retryRequest(async () => {
      const response = await this.client.post('/search/export-results', results, {
        params: { format },
        responseType: 'blob',
        timeout: 60000, // 1 minute for export
      });
      return response.data;
    });
  }

  // Get search engine statistics
  async getStatistics(): Promise<any> {
    return this.retryRequest(async () => {
      const response = await this.client.get('/search/statistics');
      return response.data;
    });
  }

  // Load saved models
  async loadModels(): Promise<{ status: string; message: string }> {
    return this.retryRequest(async () => {
      const response = await this.client.post('/search/load-models');
      return response.data;
    });
  }

  // Save current models
  async saveModels(): Promise<{ status: string; message: string }> {
    return this.retryRequest(async () => {
      const response = await this.client.post('/search/save-models');
      return response.data;
    });
  }

  // Update base URL if needed
  updateBaseURL(newBaseURL: string): void {
    this.baseURL = newBaseURL;
    this.client.defaults.baseURL = newBaseURL;
  }

  // Get current base URL
  getBaseURL(): string {
    return this.baseURL;
  }
}

// Create singleton instance
const apiClient = new ApiClient();

export default apiClient;
export { ApiClient };
