import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import FileUpload from '../FileUpload';

// Mock fetch
global.fetch = jest.fn();

// Mock file for testing
const createMockFile = (name: string, size: number, type: string) => {
  const file = new File(['test content'], name, { type });
  Object.defineProperty(file, 'size', { value: size });
  return file;
};

describe('FileUpload Component', () => {
  const mockOnUploadSuccess = jest.fn();
  const mockOnUploadError = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    (fetch as jest.Mock).mockClear();
  });

  it('renders upload area', () => {
    render(
      <FileUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    expect(screen.getByText('Upload Catalog File')).toBeInTheDocument();
    expect(screen.getByText(/drop your file here, or click to browse/i)).toBeInTheDocument();
    expect(screen.getByText(/supports csv, excel/i)).toBeInTheDocument();
  });

  it('has proper accessibility attributes', () => {
    render(
      <FileUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    const dropzone = screen.getByRole('button', { name: /upload catalog file/i });
    expect(dropzone).toHaveAttribute('tabIndex', '0');
    expect(dropzone).toHaveAttribute('aria-describedby', 'upload-description');
  });

  it('accepts valid file types', async () => {
    const user = userEvent.setup();
    (fetch as jest.Mock)
      // upload-catalog
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'queued', task_id: 'task123' })
      })
      // upload-status polling
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'success', successful: true, statistics: { dataset: { total_rows: 100, columns: [], per_column: {} }, engine: { catalog_size: 100 } } })
      });

    render(
      <FileUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    const file = createMockFile('test.csv', 1000, 'text/csv');
    const input = screen.getByLabelText(/file upload input/i);

    await user.upload(input, file);

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/search/upload-catalog',
        expect.objectContaining({
          method: 'POST',
          body: expect.any(FormData)
        })
      );
    });
  });

  it('rejects invalid file types', async () => {
    const user = userEvent.setup();
    
    render(
      <FileUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    const file = createMockFile('test.txt', 1000, 'text/plain');
    const input = screen.getByLabelText(/file upload input/i);

    await user.upload(input, file);

    await waitFor(() => {
      expect(mockOnUploadError).toHaveBeenCalledWith(
        'Only CSV, Excel (.xlsx, .xls), and JSON files are supported'
      );
    });
  });

  it('rejects files that are too large', async () => {
    const user = userEvent.setup();
    
    render(
      <FileUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    const file = createMockFile('large.csv', 60 * 1024 * 1024, 'text/csv'); // 60MB
    const input = screen.getByLabelText(/file upload input/i);

    await user.upload(input, file);

    await waitFor(() => {
      expect(mockOnUploadError).toHaveBeenCalledWith(
        'File size must be less than 50MB'
      );
    });
  });

  it('shows loading state during upload', async () => {
    const user = userEvent.setup();
    (fetch as jest.Mock).mockImplementation(() => 
      new Promise(resolve => setTimeout(() => resolve({
        ok: true,
        json: async () => ({ status: 'success', message: 'Uploaded' })
      }), 100))
    );

    render(
      <FileUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    const file = createMockFile('test.csv', 1000, 'text/csv');
    const input = screen.getByLabelText(/file upload input/i);

    await user.upload(input, file);

    expect(screen.getByText('Uploading...')).toBeInTheDocument();
    expect(screen.getByRole('status')).toBeInTheDocument();
  });

  it('displays uploaded file information', async () => {
    const user = userEvent.setup();
    (fetch as jest.Mock)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'queued', task_id: 'task123' })
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'success', successful: true, statistics: { dataset: { total_rows: 150, columns: [], per_column: {} }, engine: { catalog_size: 150 } } })
      });

    render(
      <FileUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    const file = createMockFile('catalog.xlsx', 2048, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
    const input = screen.getByLabelText(/file upload input/i);

    await user.upload(input, file);

    await waitFor(() => {
      expect(screen.getByText('catalog.xlsx')).toBeInTheDocument();
      expect(screen.getByText(/2 KB • 150 records/)).toBeInTheDocument();
    });
  });

  it('allows removing uploaded file', async () => {
    const user = userEvent.setup();
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        status: 'success',
        message: 'File uploaded successfully'
      })
    });

    render(
      <FileUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    const file = createMockFile('test.csv', 1000, 'text/csv');
    const input = screen.getByLabelText(/file upload input/i);

    await user.upload(input, file);

    await waitFor(() => {
      expect(screen.getByText('test.csv')).toBeInTheDocument();
    });

    const removeButton = screen.getByRole('button', { name: /remove/i });
    await user.click(removeButton);

    expect(screen.queryByText('test.csv')).not.toBeInTheDocument();
    expect(screen.getByText(/drop your file here/i)).toBeInTheDocument();
  });

  it('handles upload errors', async () => {
    const user = userEvent.setup();
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      json: async () => ({
        detail: 'Upload failed'
      })
    });

    render(
      <FileUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    const file = createMockFile('test.csv', 1000, 'text/csv');
    const input = screen.getByLabelText(/file upload input/i);

    await user.upload(input, file);

    await waitFor(() => {
      expect(mockOnUploadError).toHaveBeenCalledWith('Upload failed');
    });
  });

  it('handles network errors', async () => {
    const user = userEvent.setup();
    (fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

    render(
      <FileUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    const file = createMockFile('test.csv', 1000, 'text/csv');
    const input = screen.getByLabelText(/file upload input/i);

    await user.upload(input, file);

    await waitFor(() => {
      expect(mockOnUploadError).toHaveBeenCalledWith('Network error');
    });
  });

  it('is disabled when disabled prop is true', () => {
    render(
      <FileUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
        disabled
      />
    );

    const dropzone = screen.getByRole('button');
    expect(dropzone).toHaveClass('opacity-50', 'cursor-not-allowed');
  });

  it('supports drag and drop', async () => {
    render(
      <FileUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    const dropzone = screen.getByRole('button');
    const file = createMockFile('test.csv', 1000, 'text/csv');

    fireEvent.dragEnter(dropzone, {
      dataTransfer: { files: [file] }
    });

    expect(screen.getByText('Drop the file here')).toBeInTheDocument();
  });

  it('formats file size correctly', async () => {
    const user = userEvent.setup();
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: 'success', message: 'Uploaded' })
    });

    render(
      <FileUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    const file = createMockFile('test.csv', 1024 * 1024, 'text/csv'); // 1MB
    const input = screen.getByLabelText(/file upload input/i);

    await user.upload(input, file);

    await waitFor(() => {
      expect(screen.getByText(/1 MB/)).toBeInTheDocument();
    });
  });
});
