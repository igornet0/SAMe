import apiClient, { ApiClient } from '../apiClient';

// Mock axios
jest.mock('axios');
import axios from 'axios';
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('ApiClient', () => {
  let client: ApiClient;

  beforeEach(() => {
    client = new ApiClient('http://localhost:8000');
    jest.clearAllMocks();
  });

  describe('constructor', () => {
    it('creates client with default base URL', () => {
      const defaultClient = new ApiClient();
      expect(defaultClient.getBaseURL()).toBe('http://localhost:8000');
    });

    it('creates client with custom base URL', () => {
      const customClient = new ApiClient('http://custom:9000');
      expect(customClient.getBaseURL()).toBe('http://custom:9000');
    });
  });

  describe('healthCheck', () => {
    it('performs health check successfully', async () => {
      const mockResponse = { status: 'active', message: 'SAMe API' };
      mockedAxios.create.mockReturnValue({
        get: jest.fn().mockResolvedValue({ data: mockResponse }),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() }
        }
      } as any);

      const result = await client.healthCheck();
      expect(result).toEqual(mockResponse);
    });

    it('retries on failure', async () => {
      const mockAxiosInstance = {
        get: jest.fn()
          .mockRejectedValueOnce(new Error('Network error'))
          .mockRejectedValueOnce(new Error('Network error'))
          .mockResolvedValueOnce({ data: { status: 'active' } }),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() }
        }
      };

      mockedAxios.create.mockReturnValue(mockAxiosInstance as any);

      const result = await client.healthCheck();
      expect(result).toEqual({ status: 'active' });
      expect(mockAxiosInstance.get).toHaveBeenCalledTimes(3);
    });
  });

  describe('uploadCatalog', () => {
    it('uploads file successfully', async () => {
      const mockFile = new File(['test'], 'test.csv', { type: 'text/csv' });
      const mockResponse = {
        status: 'success',
        message: 'File uploaded',
        statistics: { total_items: 100 }
      };

      const mockAxiosInstance = {
        post: jest.fn().mockResolvedValue({ data: mockResponse }),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() }
        }
      };

      mockedAxios.create.mockReturnValue(mockAxiosInstance as any);

      const result = await client.uploadCatalog(mockFile);
      
      expect(result).toEqual(mockResponse);
      expect(mockAxiosInstance.post).toHaveBeenCalledWith(
        '/search/upload-catalog',
        expect.any(FormData),
        expect.objectContaining({
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 60000
        })
      );
    });

    it('handles upload errors', async () => {
      const mockFile = new File(['test'], 'test.csv', { type: 'text/csv' });
      const mockAxiosInstance = {
        post: jest.fn().mockRejectedValue({
          response: {
            status: 400,
            data: { detail: 'Invalid file format' }
          }
        }),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() }
        }
      };

      mockedAxios.create.mockReturnValue(mockAxiosInstance as any);

      await expect(client.uploadCatalog(mockFile)).rejects.toEqual({
        detail: 'Invalid file format',
        status_code: 400
      });
    });
  });

  describe('searchAnalogs', () => {
    it('searches for analogs successfully', async () => {
      const mockRequest = {
        queries: ['болт м10'],
        method: 'hybrid',
        similarity_threshold: 0.6,
        max_results: 10
      };

      const mockResponse = {
        results: { 'болт м10': [{ document: 'Болт М10х50', similarity_score: 0.95 }] },
        statistics: {},
        processing_time: 0.5
      };

      const mockAxiosInstance = {
        post: jest.fn().mockResolvedValue({ data: mockResponse }),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() }
        }
      };

      mockedAxios.create.mockReturnValue(mockAxiosInstance as any);

      const result = await client.searchAnalogs(mockRequest);
      
      expect(result).toEqual(mockResponse);
      expect(mockAxiosInstance.post).toHaveBeenCalledWith(
        '/search/search-analogs',
        mockRequest
      );
    });

    it('handles search errors', async () => {
      const mockRequest = {
        queries: ['test'],
        method: 'hybrid'
      };

      const mockAxiosInstance = {
        post: jest.fn().mockRejectedValue({
          response: {
            status: 500,
            data: { detail: 'Search engine not initialized' }
          }
        }),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() }
        }
      };

      mockedAxios.create.mockReturnValue(mockAxiosInstance as any);

      await expect(client.searchAnalogs(mockRequest)).rejects.toEqual({
        detail: 'Search engine not initialized',
        status_code: 500
      });
    });
  });

  describe('exportResults', () => {
    it('exports results successfully', async () => {
      const mockResults = { 'query': [{ document: 'result' }] };
      const mockBlob = new Blob(['excel data'], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });

      const mockAxiosInstance = {
        post: jest.fn().mockResolvedValue({ data: mockBlob }),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() }
        }
      };

      mockedAxios.create.mockReturnValue(mockAxiosInstance as any);

      const result = await client.exportResults(mockResults);
      
      expect(result).toEqual(mockBlob);
      expect(mockAxiosInstance.post).toHaveBeenCalledWith(
        '/search/export-results',
        mockResults,
        expect.objectContaining({
          params: { format: 'excel' },
          responseType: 'blob',
          timeout: 60000
        })
      );
    });
  });

  describe('error handling', () => {
    it('handles network errors', async () => {
      const mockAxiosInstance = {
        get: jest.fn().mockRejectedValue({
          request: {},
          message: 'Network Error'
        }),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() }
        }
      };

      mockedAxios.create.mockReturnValue(mockAxiosInstance as any);

      await expect(client.healthCheck()).rejects.toEqual({
        detail: 'Network error: Unable to connect to server. Please check your connection.',
        status_code: 0
      });
    });

    it('handles generic errors', async () => {
      const mockAxiosInstance = {
        get: jest.fn().mockRejectedValue(new Error('Generic error')),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() }
        }
      };

      mockedAxios.create.mockReturnValue(mockAxiosInstance as any);

      await expect(client.healthCheck()).rejects.toEqual({
        detail: 'Generic error',
        status_code: 0
      });
    });
  });

  describe('retry logic', () => {
    it('retries on server errors', async () => {
      const mockAxiosInstance = {
        get: jest.fn()
          .mockRejectedValueOnce({
            response: { status: 500, data: { detail: 'Server error' } }
          })
          .mockResolvedValueOnce({ data: { status: 'active' } }),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() }
        }
      };

      mockedAxios.create.mockReturnValue(mockAxiosInstance as any);

      const result = await client.healthCheck();
      expect(result).toEqual({ status: 'active' });
      expect(mockAxiosInstance.get).toHaveBeenCalledTimes(2);
    });

    it('does not retry on client errors', async () => {
      const mockAxiosInstance = {
        get: jest.fn().mockRejectedValue({
          response: { status: 400, data: { detail: 'Bad request' } }
        }),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() }
        }
      };

      mockedAxios.create.mockReturnValue(mockAxiosInstance as any);

      await expect(client.healthCheck()).rejects.toEqual({
        detail: 'Bad request',
        status_code: 400
      });
      expect(mockAxiosInstance.get).toHaveBeenCalledTimes(1);
    });
  });

  describe('URL management', () => {
    it('updates base URL', () => {
      client.updateBaseURL('http://new-url:8080');
      expect(client.getBaseURL()).toBe('http://new-url:8080');
    });
  });

  describe('singleton instance', () => {
    it('exports singleton instance', () => {
      expect(apiClient).toBeInstanceOf(ApiClient);
      expect(apiClient.getBaseURL()).toBe('http://localhost:8000');
    });
  });
});
