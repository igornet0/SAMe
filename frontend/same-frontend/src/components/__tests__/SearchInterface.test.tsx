import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import SearchInterface from '../SearchInterface';

// Mock fetch
global.fetch = jest.fn();

describe('SearchInterface Component', () => {
  const mockOnSearchResults = jest.fn();
  const mockOnSearchError = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    (fetch as jest.Mock).mockClear();
  });

  it('renders search interface', () => {
    render(
      <SearchInterface
        catalogUploaded={true}
        onSearchResults={mockOnSearchResults}
        onSearchError={mockOnSearchError}
      />
    );

    expect(screen.getByText('Search for Analogs')).toBeInTheDocument();
    expect(screen.getByLabelText(/product name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/search method/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /search analogs/i })).toBeInTheDocument();
  });

  it('shows warning when catalog is not uploaded', () => {
    render(
      <SearchInterface
        catalogUploaded={false}
        onSearchResults={mockOnSearchResults}
        onSearchError={mockOnSearchError}
      />
    );

    expect(screen.getByText(/please upload a catalog file first/i)).toBeInTheDocument();
  });

  it('disables search when catalog is not uploaded', () => {
    render(
      <SearchInterface
        catalogUploaded={false}
        onSearchResults={mockOnSearchResults}
        onSearchError={mockOnSearchError}
      />
    );

    const searchButton = screen.getByRole('button', { name: /search analogs/i });
    const productInput = screen.getByLabelText(/product name/i);
    const methodSelect = screen.getByLabelText(/search method/i);

    expect(searchButton).toBeDisabled();
    expect(productInput).toBeDisabled();
    expect(methodSelect).toBeDisabled();
  });

  it('validates empty search query', async () => {
    const user = userEvent.setup();
    
    render(
      <SearchInterface
        catalogUploaded={true}
        onSearchResults={mockOnSearchResults}
        onSearchError={mockOnSearchError}
      />
    );

    const searchButton = screen.getByRole('button', { name: /search analogs/i });
    await user.click(searchButton);

    await waitFor(() => {
      expect(mockOnSearchError).toHaveBeenCalledWith(
        'Please enter a product name to search for'
      );
    });
  });

  it('validates minimum query length', async () => {
    const user = userEvent.setup();
    
    render(
      <SearchInterface
        catalogUploaded={true}
        onSearchResults={mockOnSearchResults}
        onSearchError={mockOnSearchError}
      />
    );

    const productInput = screen.getByLabelText(/product name/i);
    const searchButton = screen.getByRole('button', { name: /search analogs/i });

    await user.type(productInput, 'a');
    await user.click(searchButton);

    await waitFor(() => {
      expect(mockOnSearchError).toHaveBeenCalledWith(
        'Search query must be at least 2 characters long'
      );
    });
  });

  it('performs successful search', async () => {
    const user = userEvent.setup();
    const mockResults = [
      {
        document_id: '1',
        document: 'Болт М10х50 ГОСТ 7798-70',
        similarity_score: 0.95,
        rank: 1
      }
    ];

    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        results: { 'болт м10': mockResults },
        statistics: {},
        processing_time: 0.5
      })
    });

    render(
      <SearchInterface
        catalogUploaded={true}
        onSearchResults={mockOnSearchResults}
        onSearchError={mockOnSearchError}
      />
    );

    const productInput = screen.getByLabelText(/product name/i);
    const searchButton = screen.getByRole('button', { name: /search analogs/i });

    await user.type(productInput, 'болт м10');
    await user.click(searchButton);

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/search/search-analogs',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            queries: ['болт м10'],
            method: 'hybrid',
            similarity_threshold: 0.6,
            max_results: 10
          })
        })
      );
      expect(mockOnSearchResults).toHaveBeenCalledWith(mockResults, 'болт м10');
    });
  });

  it('shows loading state during search', async () => {
    const user = userEvent.setup();
    (fetch as jest.Mock).mockImplementation(() => 
      new Promise(resolve => setTimeout(() => resolve({
        ok: true,
        json: async () => ({ results: {}, statistics: {}, processing_time: 0.5 })
      }), 100))
    );

    render(
      <SearchInterface
        catalogUploaded={true}
        onSearchResults={mockOnSearchResults}
        onSearchError={mockOnSearchError}
      />
    );

    const productInput = screen.getByLabelText(/product name/i);
    const searchButton = screen.getByRole('button', { name: /search analogs/i });

    await user.type(productInput, 'test query');
    await user.click(searchButton);

    expect(screen.getByText('Searching...')).toBeInTheDocument();
    expect(screen.getByText(/searching for analogs/i)).toBeInTheDocument();
  });

  it('handles search errors', async () => {
    const user = userEvent.setup();
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      json: async () => ({
        detail: 'Search failed'
      })
    });

    render(
      <SearchInterface
        catalogUploaded={true}
        onSearchResults={mockOnSearchResults}
        onSearchError={mockOnSearchError}
      />
    );

    const productInput = screen.getByLabelText(/product name/i);
    const searchButton = screen.getByRole('button', { name: /search analogs/i });

    await user.type(productInput, 'test query');
    await user.click(searchButton);

    await waitFor(() => {
      expect(mockOnSearchError).toHaveBeenCalledWith('Search failed');
    });
  });

  it('handles network errors', async () => {
    const user = userEvent.setup();
    (fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

    render(
      <SearchInterface
        catalogUploaded={true}
        onSearchResults={mockOnSearchResults}
        onSearchError={mockOnSearchError}
      />
    );

    const productInput = screen.getByLabelText(/product name/i);
    const searchButton = screen.getByRole('button', { name: /search analogs/i });

    await user.type(productInput, 'test query');
    await user.click(searchButton);

    await waitFor(() => {
      expect(mockOnSearchError).toHaveBeenCalledWith('Network error');
    });
  });

  it('supports different search methods', async () => {
    const user = userEvent.setup();
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ results: {}, statistics: {}, processing_time: 0.5 })
    });

    render(
      <SearchInterface
        catalogUploaded={true}
        onSearchResults={mockOnSearchResults}
        onSearchError={mockOnSearchError}
      />
    );

    const methodSelect = screen.getByLabelText(/search method/i);
    const productInput = screen.getByLabelText(/product name/i);
    const searchButton = screen.getByRole('button', { name: /search analogs/i });

    await user.selectOptions(methodSelect, 'semantic');
    await user.type(productInput, 'test query');
    await user.click(searchButton);

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/search/search-analogs',
        expect.objectContaining({
          body: JSON.stringify(expect.objectContaining({
            method: 'semantic'
          }))
        })
      );
    });
  });

  it('supports form submission via Enter key', async () => {
    const user = userEvent.setup();
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ results: {}, statistics: {}, processing_time: 0.5 })
    });

    render(
      <SearchInterface
        catalogUploaded={true}
        onSearchResults={mockOnSearchResults}
        onSearchError={mockOnSearchError}
      />
    );

    const productInput = screen.getByLabelText(/product name/i);
    await user.type(productInput, 'test query{enter}');

    await waitFor(() => {
      expect(fetch).toHaveBeenCalled();
    });
  });

  it('trims whitespace from search query', async () => {
    const user = userEvent.setup();
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ results: { 'test query': [] }, statistics: {}, processing_time: 0.5 })
    });

    render(
      <SearchInterface
        catalogUploaded={true}
        onSearchResults={mockOnSearchResults}
        onSearchError={mockOnSearchError}
      />
    );

    const productInput = screen.getByLabelText(/product name/i);
    const searchButton = screen.getByRole('button', { name: /search analogs/i });

    await user.type(productInput, '  test query  ');
    await user.click(searchButton);

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/search/search-analogs',
        expect.objectContaining({
          body: JSON.stringify(expect.objectContaining({
            queries: ['test query']
          }))
        })
      );
    });
  });

  it('has proper accessibility attributes', () => {
    render(
      <SearchInterface
        catalogUploaded={true}
        onSearchResults={mockOnSearchResults}
        onSearchError={mockOnSearchError}
      />
    );

    const productInput = screen.getByLabelText(/product name/i);
    const methodSelect = screen.getByLabelText(/search method/i);

    expect(productInput).toHaveAttribute('aria-required', 'true');
    expect(methodSelect).toHaveAttribute('aria-describedby', 'search-method-help');
  });

  it('is disabled when disabled prop is true', () => {
    render(
      <SearchInterface
        catalogUploaded={true}
        onSearchResults={mockOnSearchResults}
        onSearchError={mockOnSearchError}
        disabled
      />
    );

    const searchButton = screen.getByRole('button', { name: /search analogs/i });
    const productInput = screen.getByLabelText(/product name/i);
    const methodSelect = screen.getByLabelText(/search method/i);

    expect(searchButton).toBeDisabled();
    expect(productInput).toBeDisabled();
    expect(methodSelect).toBeDisabled();
  });
});
