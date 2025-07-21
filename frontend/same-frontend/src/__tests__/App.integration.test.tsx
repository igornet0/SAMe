import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import App from '../App';

// Mock fetch
global.fetch = jest.fn();

// Mock URL.createObjectURL and URL.revokeObjectURL
global.URL.createObjectURL = jest.fn(() => 'mock-url');
global.URL.revokeObjectURL = jest.fn();

// Mock file for testing
const createMockFile = (name: string, size: number, type: string) => {
  const file = new File(['test content'], name, { type });
  Object.defineProperty(file, 'size', { value: size });
  return file;
};

describe('App Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (fetch as jest.Mock).mockClear();
  });

  it('completes full user workflow: upload -> search -> export', async () => {
    const user = userEvent.setup();

    // Mock API responses
    (fetch as jest.Mock)
      // Health check
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'active', message: 'SAMe API' })
      })
      // File upload
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: 'success',
          message: 'Catalog uploaded successfully',
          statistics: { total_items: 150 }
        })
      })
      // Search
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          results: {
            'болт м10': [
              {
                document_id: '1',
                document: 'Болт М10х50 ГОСТ 7798-70',
                similarity_score: 0.95,
                rank: 1
              },
              {
                document_id: '2',
                document: 'Болт М10х40 DIN 912',
                similarity_score: 0.87,
                rank: 2
              }
            ]
          },
          statistics: {},
          processing_time: 0.8
        })
      })
      // Export
      .mockResolvedValueOnce({
        ok: true,
        blob: async () => new Blob(['excel data'], { 
          type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' 
        })
      });

    render(<App />);

    // Wait for app to load and connect to backend
    await waitFor(() => {
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });

    // Step 1: Upload catalog file
    const file = createMockFile('catalog.xlsx', 2048, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
    const fileInput = screen.getByLabelText(/file upload input/i);
    
    await user.upload(fileInput, file);

    // Wait for upload to complete
    await waitFor(() => {
      expect(screen.getByText('catalog.xlsx')).toBeInTheDocument();
    });

    // Step 2: Search for analogs
    const productInput = screen.getByLabelText(/product name/i);
    const searchButton = screen.getByRole('button', { name: /search analogs/i });

    await user.type(productInput, 'болт м10');
    await user.click(searchButton);

    // Wait for search results
    await waitFor(() => {
      expect(screen.getByText('Болт М10х50 ГОСТ 7798-70')).toBeInTheDocument();
      expect(screen.getByText('Болт М10х40 DIN 912')).toBeInTheDocument();
    });

    // Step 3: Export results
    const exportButton = screen.getByRole('button', { name: /export to excel/i });
    await user.click(exportButton);

    // Verify all API calls were made
    expect(fetch).toHaveBeenCalledTimes(4);
    
    // Verify health check
    expect(fetch).toHaveBeenNthCalledWith(1, 'http://localhost:8000/search/', expect.any(Object));
    
    // Verify file upload
    expect(fetch).toHaveBeenNthCalledWith(2, 'http://localhost:8000/search/upload-catalog', expect.objectContaining({
      method: 'POST',
      body: expect.any(FormData)
    }));
    
    // Verify search
    expect(fetch).toHaveBeenNthCalledWith(3, 'http://localhost:8000/search/search-analogs', expect.objectContaining({
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        queries: ['болт м10'],
        method: 'hybrid',
        similarity_threshold: 0.6,
        max_results: 10
      })
    }));
    
    // Verify export
    expect(fetch).toHaveBeenNthCalledWith(4, 'http://localhost:8000/search/export-results', expect.objectContaining({
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    }));
  });

  it('handles backend connection failure', async () => {
    (fetch as jest.Mock).mockRejectedValue(new Error('Connection failed'));

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/backend connection failed/i)).toBeInTheDocument();
      expect(screen.getByText(/unable to connect to the same backend server/i)).toBeInTheDocument();
    });

    // Should show retry button
    expect(screen.getByText('Try Again')).toBeInTheDocument();
  });

  it('shows error when searching without uploading catalog', async () => {
    const user = userEvent.setup();

    // Mock successful health check
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: 'active', message: 'SAMe API' })
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });

    // Try to search without uploading catalog
    const productInput = screen.getByLabelText(/product name/i);
    const searchButton = screen.getByRole('button', { name: /search analogs/i });

    // Input should be disabled
    expect(productInput).toBeDisabled();
    expect(searchButton).toBeDisabled();

    // Should show warning message
    expect(screen.getByText(/please upload a catalog file first/i)).toBeInTheDocument();
  });

  it('handles file upload errors gracefully', async () => {
    const user = userEvent.setup();

    // Mock health check and failed upload
    (fetch as jest.Mock)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'active', message: 'SAMe API' })
      })
      .mockResolvedValueOnce({
        ok: false,
        json: async () => ({ detail: 'Invalid file format' })
      });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });

    // Try to upload invalid file
    const file = createMockFile('invalid.txt', 1000, 'text/plain');
    const fileInput = screen.getByLabelText(/file upload input/i);
    
    await user.upload(fileInput, file);

    // Should show error message
    await waitFor(() => {
      expect(screen.getByText(/only csv, excel/i)).toBeInTheDocument();
    });
  });

  it('handles search errors gracefully', async () => {
    const user = userEvent.setup();

    // Mock successful health check and upload, failed search
    (fetch as jest.Mock)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'active', message: 'SAMe API' })
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: 'success',
          message: 'Catalog uploaded successfully',
          statistics: { total_items: 100 }
        })
      })
      .mockResolvedValueOnce({
        ok: false,
        json: async () => ({ detail: 'Search engine error' })
      });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });

    // Upload file
    const file = createMockFile('catalog.csv', 1000, 'text/csv');
    const fileInput = screen.getByLabelText(/file upload input/i);
    await user.upload(fileInput, file);

    await waitFor(() => {
      expect(screen.getByText('catalog.csv')).toBeInTheDocument();
    });

    // Try to search
    const productInput = screen.getByLabelText(/product name/i);
    const searchButton = screen.getByRole('button', { name: /search analogs/i });

    await user.type(productInput, 'test query');
    await user.click(searchButton);

    // Should show error message
    await waitFor(() => {
      expect(screen.getByText('Search engine error')).toBeInTheDocument();
    });
  });

  it('shows empty state when no search results found', async () => {
    const user = userEvent.setup();

    // Mock successful health check, upload, and empty search results
    (fetch as jest.Mock)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'active', message: 'SAMe API' })
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: 'success',
          message: 'Catalog uploaded successfully',
          statistics: { total_items: 100 }
        })
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          results: { 'nonexistent item': [] },
          statistics: {},
          processing_time: 0.3
        })
      });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });

    // Upload file
    const file = createMockFile('catalog.csv', 1000, 'text/csv');
    const fileInput = screen.getByLabelText(/file upload input/i);
    await user.upload(fileInput, file);

    await waitFor(() => {
      expect(screen.getByText('catalog.csv')).toBeInTheDocument();
    });

    // Search for non-existent item
    const productInput = screen.getByLabelText(/product name/i);
    const searchButton = screen.getByRole('button', { name: /search analogs/i });

    await user.type(productInput, 'nonexistent item');
    await user.click(searchButton);

    // Should show empty state
    await waitFor(() => {
      expect(screen.getByText('No results found')).toBeInTheDocument();
      expect(screen.getByText(/no analogs were found for "nonexistent item"/i)).toBeInTheDocument();
    });
  });

  it('maintains responsive design on different screen sizes', async () => {
    // Mock successful health check
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: 'active', message: 'SAMe API' })
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });

    // Check that responsive classes are applied
    const header = screen.getByRole('banner');
    expect(header.querySelector('.max-w-7xl')).toBeInTheDocument();
    expect(header.querySelector('.px-4.sm\\:px-6.lg\\:px-8')).toBeInTheDocument();

    const main = screen.getByRole('main');
    expect(main).toHaveClass('max-w-7xl', 'mx-auto');
  });
});
