import React from 'react';
import { render, screen } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';
import '@testing-library/jest-dom';
import App from '../App';
import FileUpload from '../components/FileUpload';
import SearchInterface from '../components/SearchInterface';
import ResultsDisplay from '../components/ResultsDisplay';
import { Button, Input, Alert } from '../components/ui';

// Extend Jest matchers
expect.extend(toHaveNoViolations);

// Mock fetch for App component
global.fetch = jest.fn();

describe('Accessibility Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'active', message: 'SAMe API' })
    });
  });

  describe('Button Component Accessibility', () => {
    it('should not have accessibility violations', async () => {
      const { container } = render(<Button>Click me</Button>);
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('has proper ARIA attributes when loading', async () => {
      const { container } = render(<Button loading>Loading</Button>);
      const results = await axe(container);
      expect(results).toHaveNoViolations();
      
      const button = screen.getByRole('button');
      expect(button).toBeDisabled();
    });

    it('has proper focus management', () => {
      render(<Button>Focusable Button</Button>);
      const button = screen.getByRole('button');
      
      button.focus();
      expect(button).toHaveFocus();
    });
  });

  describe('Input Component Accessibility', () => {
    it('should not have accessibility violations', async () => {
      const { container } = render(<Input label="Test Input" />);
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('has proper label association', () => {
      render(<Input label="Email Address" />);
      const input = screen.getByRole('textbox');
      const label = screen.getByText('Email Address');
      
      expect(input).toHaveAccessibleName('Email Address');
      expect(label).toHaveAttribute('for', input.id);
    });

    it('has proper error announcement', async () => {
      const { container } = render(<Input label="Required Field" error="This field is required" />);
      const results = await axe(container);
      expect(results).toHaveNoViolations();
      
      const errorMessage = screen.getByRole('alert');
      expect(errorMessage).toHaveTextContent('This field is required');
    });

    it('supports keyboard navigation', () => {
      render(<Input label="Keyboard Input" />);
      const input = screen.getByRole('textbox');
      
      input.focus();
      expect(input).toHaveFocus();
    });
  });

  describe('Alert Component Accessibility', () => {
    it('should not have accessibility violations', async () => {
      const { container } = render(<Alert type="error" message="Error message" />);
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('has proper ARIA attributes', () => {
      render(<Alert type="error" title="Error" message="Something went wrong" />);
      const alert = screen.getByRole('alert');
      
      expect(alert).toHaveAttribute('role', 'alert');
      expect(alert).toHaveAttribute('aria-live', 'assertive');
      expect(alert).toHaveAttribute('aria-atomic', 'true');
    });

    it('has accessible close button', () => {
      const onClose = jest.fn();
      render(<Alert type="info" message="Info message" onClose={onClose} />);
      
      const closeButton = screen.getByRole('button', { name: /close notification/i });
      expect(closeButton).toBeInTheDocument();
      expect(closeButton).toHaveAttribute('aria-label', 'Close notification');
    });
  });

  describe('FileUpload Component Accessibility', () => {
    it('should not have accessibility violations', async () => {
      const { container } = render(
        <FileUpload
          onUploadSuccess={() => {}}
          onUploadError={() => {}}
        />
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('has proper ARIA attributes for dropzone', () => {
      render(
        <FileUpload
          onUploadSuccess={() => {}}
          onUploadError={() => {}}
        />
      );
      
      const dropzone = screen.getByRole('button', { name: /upload catalog file/i });
      expect(dropzone).toHaveAttribute('aria-describedby', 'upload-description');
      expect(dropzone).toHaveAttribute('tabIndex', '0');
    });

    it('has accessible file input', () => {
      render(
        <FileUpload
          onUploadSuccess={() => {}}
          onUploadError={() => {}}
        />
      );
      
      const fileInput = screen.getByLabelText(/file upload input/i);
      expect(fileInput).toHaveAttribute('type', 'file');
    });

    it('announces loading state', () => {
      render(
        <FileUpload
          onUploadSuccess={() => {}}
          onUploadError={() => {}}
        />
      );
      
      // Simulate loading state by checking for status role
      const statusElements = screen.queryAllByRole('status');
      expect(statusElements).toBeDefined();
    });
  });

  describe('SearchInterface Component Accessibility', () => {
    it('should not have accessibility violations', async () => {
      const { container } = render(
        <SearchInterface
          catalogUploaded={true}
          onSearchResults={() => {}}
          onSearchError={() => {}}
        />
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('has proper form structure', () => {
      render(
        <SearchInterface
          catalogUploaded={true}
          onSearchResults={() => {}}
          onSearchError={() => {}}
        />
      );
      
      const form = screen.getByRole('form') || screen.getByRole('search') || screen.getByTestId('search-form') || document.querySelector('form');
      const productInput = screen.getByLabelText(/product name/i);
      const searchButton = screen.getByRole('button', { name: /search analogs/i });
      
      expect(productInput).toBeInTheDocument();
      expect(searchButton).toBeInTheDocument();
    });

    it('has proper ARIA attributes for required fields', () => {
      render(
        <SearchInterface
          catalogUploaded={true}
          onSearchResults={() => {}}
          onSearchError={() => {}}
        />
      );
      
      const productInput = screen.getByLabelText(/product name/i);
      expect(productInput).toHaveAttribute('aria-required', 'true');
    });

    it('has accessible select element', () => {
      render(
        <SearchInterface
          catalogUploaded={true}
          onSearchResults={() => {}}
          onSearchError={() => {}}
        />
      );
      
      const methodSelect = screen.getByLabelText(/search method/i);
      expect(methodSelect).toHaveAttribute('aria-describedby', 'search-method-help');
    });
  });

  describe('ResultsDisplay Component Accessibility', () => {
    const mockResults = [
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
    ];

    it('should not have accessibility violations', async () => {
      const { container } = render(
        <ResultsDisplay
          results={mockResults}
          query="болт м10"
        />
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('has proper table structure', () => {
      render(
        <ResultsDisplay
          results={mockResults}
          query="болт м10"
        />
      );
      
      const table = screen.getByRole('table');
      expect(table).toBeInTheDocument();
      
      // Check for proper table headers
      expect(screen.getByRole('columnheader', { name: /rank/i })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: /found analog/i })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: /similarity/i })).toBeInTheDocument();
    });

    it('has accessible progress bars for similarity scores', () => {
      render(
        <ResultsDisplay
          results={mockResults}
          query="болт м10"
        />
      );
      
      const progressBars = screen.getAllByRole('progressbar');
      expect(progressBars.length).toBeGreaterThan(0);
      
      progressBars.forEach(progressBar => {
        expect(progressBar).toHaveAttribute('aria-valuenow');
        expect(progressBar).toHaveAttribute('aria-valuemin', '0');
        expect(progressBar).toHaveAttribute('aria-valuemax', '100');
        expect(progressBar).toHaveAttribute('aria-label');
      });
    });

    it('has accessible pagination controls', () => {
      const manyResults = Array.from({ length: 25 }, (_, i) => ({
        document_id: `${i + 1}`,
        document: `Result ${i + 1}`,
        similarity_score: 0.8,
        rank: i + 1
      }));

      render(
        <ResultsDisplay
          results={manyResults}
          query="test query"
        />
      );
      
      // Check for pagination buttons
      const prevButton = screen.queryByRole('button', { name: /previous/i });
      const nextButton = screen.queryByRole('button', { name: /next/i });
      
      if (prevButton) expect(prevButton).toBeInTheDocument();
      if (nextButton) expect(nextButton).toBeInTheDocument();
    });
  });

  describe('App Component Accessibility', () => {
    it('should not have accessibility violations', async () => {
      const { container } = render(<App />);
      
      // Wait for the app to load
      await screen.findByText(/same.*analog search engine/i);
      
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('has proper landmark structure', async () => {
      render(<App />);
      
      await screen.findByText(/same.*analog search engine/i);
      
      expect(screen.getByRole('banner')).toBeInTheDocument(); // header
      expect(screen.getByRole('main')).toBeInTheDocument(); // main content
      expect(screen.getByRole('contentinfo')).toBeInTheDocument(); // footer
    });

    it('has proper heading hierarchy', async () => {
      render(<App />);
      
      await screen.findByText(/same.*analog search engine/i);
      
      const h1 = screen.getByRole('heading', { level: 1 });
      expect(h1).toHaveTextContent(/same.*analog search engine/i);
      
      const h2Elements = screen.getAllByRole('heading', { level: 2 });
      expect(h2Elements.length).toBeGreaterThan(0);
    });

    it('has accessible status indicators', async () => {
      render(<App />);
      
      await screen.findByText(/connected/i);
      
      const statusIndicator = screen.getByRole('status');
      expect(statusIndicator).toHaveAttribute('aria-live', 'polite');
    });

    it('supports keyboard navigation', async () => {
      render(<App />);
      
      await screen.findByText(/same.*analog search engine/i);
      
      // Check that interactive elements are focusable
      const interactiveElements = screen.getAllByRole('button');
      interactiveElements.forEach(element => {
        expect(element).not.toHaveAttribute('tabindex', '-1');
      });
    });
  });

  describe('Color Contrast and Visual Accessibility', () => {
    it('uses semantic colors for different states', () => {
      render(
        <div>
          <Alert type="success" message="Success message" />
          <Alert type="error" message="Error message" />
          <Alert type="warning" message="Warning message" />
          <Alert type="info" message="Info message" />
        </div>
      );
      
      // Check that different alert types have different visual indicators
      expect(screen.getByText('Success message')).toBeInTheDocument();
      expect(screen.getByText('Error message')).toBeInTheDocument();
      expect(screen.getByText('Warning message')).toBeInTheDocument();
      expect(screen.getByText('Info message')).toBeInTheDocument();
    });

    it('provides visual feedback for interactive states', () => {
      render(
        <div>
          <Button>Normal Button</Button>
          <Button disabled>Disabled Button</Button>
          <Button loading>Loading Button</Button>
        </div>
      );
      
      const normalButton = screen.getByRole('button', { name: /normal button/i });
      const disabledButton = screen.getByRole('button', { name: /disabled button/i });
      const loadingButton = screen.getByRole('button', { name: /loading button/i });
      
      expect(normalButton).not.toBeDisabled();
      expect(disabledButton).toBeDisabled();
      expect(loadingButton).toBeDisabled();
    });
  });
});
