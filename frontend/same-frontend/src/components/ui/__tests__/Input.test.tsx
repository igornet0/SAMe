import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import Input from '../Input';

describe('Input Component', () => {
  it('renders basic input', () => {
    render(<Input />);
    const input = screen.getByRole('textbox');
    
    expect(input).toBeInTheDocument();
    expect(input).toHaveClass('block', 'w-full', 'px-3', 'py-2', 'border', 'rounded-md');
  });

  it('renders with label', () => {
    render(<Input label="Test Label" />);
    const label = screen.getByText('Test Label');
    const input = screen.getByRole('textbox');
    
    expect(label).toBeInTheDocument();
    expect(label).toHaveAttribute('for', input.id);
  });

  it('renders with placeholder', () => {
    render(<Input placeholder="Enter text here" />);
    const input = screen.getByPlaceholderText('Enter text here');
    
    expect(input).toBeInTheDocument();
  });

  it('renders with error state', () => {
    render(<Input error="This field is required" />);
    const input = screen.getByRole('textbox');
    const errorMessage = screen.getByRole('alert');
    
    expect(input).toHaveClass('border-red-300', 'text-red-900');
    expect(errorMessage).toHaveTextContent('This field is required');
    expect(errorMessage).toHaveClass('text-red-600');
  });

  it('renders with helper text', () => {
    render(<Input helperText="This is helper text" />);
    const helperText = screen.getByText('This is helper text');
    
    expect(helperText).toBeInTheDocument();
    expect(helperText).toHaveClass('text-gray-500');
  });

  it('prioritizes error over helper text', () => {
    render(<Input error="Error message" helperText="Helper text" />);
    
    expect(screen.getByText('Error message')).toBeInTheDocument();
    expect(screen.queryByText('Helper text')).not.toBeInTheDocument();
  });

  it('handles value changes', async () => {
    const user = userEvent.setup();
    const handleChange = jest.fn();
    
    render(<Input onChange={handleChange} />);
    const input = screen.getByRole('textbox');
    
    await user.type(input, 'test value');
    
    expect(handleChange).toHaveBeenCalled();
    expect(input).toHaveValue('test value');
  });

  it('handles controlled input', () => {
    const { rerender } = render(<Input value="initial" onChange={() => {}} />);
    const input = screen.getByRole('textbox');
    
    expect(input).toHaveValue('initial');
    
    rerender(<Input value="updated" onChange={() => {}} />);
    expect(input).toHaveValue('updated');
  });

  it('is disabled when disabled prop is true', () => {
    render(<Input disabled />);
    const input = screen.getByRole('textbox');
    
    expect(input).toBeDisabled();
  });

  it('applies custom className', () => {
    render(<Input className="custom-class" />);
    const input = screen.getByRole('textbox');
    
    expect(input).toHaveClass('custom-class');
  });

  it('forwards other props to input element', () => {
    render(<Input data-testid="custom-input" maxLength={10} />);
    const input = screen.getByTestId('custom-input');
    
    expect(input).toHaveAttribute('maxLength', '10');
  });

  it('generates unique id when not provided', () => {
    const { rerender } = render(<Input label="First" />);
    const firstInput = screen.getByRole('textbox');
    const firstId = firstInput.id;
    
    rerender(<Input label="Second" />);
    const secondInput = screen.getByRole('textbox');
    const secondId = secondInput.id;
    
    expect(firstId).toBeTruthy();
    expect(secondId).toBeTruthy();
    expect(firstId).not.toBe(secondId);
  });

  it('uses provided id', () => {
    render(<Input id="custom-id" label="Custom ID" />);
    const input = screen.getByRole('textbox');
    const label = screen.getByText('Custom ID');
    
    expect(input).toHaveAttribute('id', 'custom-id');
    expect(label).toHaveAttribute('for', 'custom-id');
  });

  it('has proper accessibility attributes', () => {
    render(<Input label="Accessible Input" error="Error message" />);
    const input = screen.getByRole('textbox');
    const errorMessage = screen.getByRole('alert');
    
    expect(input).toHaveAccessibleName('Accessible Input');
    expect(errorMessage).toHaveAttribute('role', 'alert');
  });

  it('supports different input types', () => {
    render(<Input type="email" />);
    const input = screen.getByRole('textbox');
    
    expect(input).toHaveAttribute('type', 'email');
  });

  it('handles focus and blur events', async () => {
    const user = userEvent.setup();
    const handleFocus = jest.fn();
    const handleBlur = jest.fn();
    
    render(<Input onFocus={handleFocus} onBlur={handleBlur} />);
    const input = screen.getByRole('textbox');
    
    await user.click(input);
    expect(handleFocus).toHaveBeenCalledTimes(1);
    
    await user.tab();
    expect(handleBlur).toHaveBeenCalledTimes(1);
  });

  it('handles keyboard events', async () => {
    const user = userEvent.setup();
    const handleKeyPress = jest.fn();
    
    render(<Input onKeyPress={handleKeyPress} />);
    const input = screen.getByRole('textbox');
    
    await user.type(input, 'a');
    expect(handleKeyPress).toHaveBeenCalled();
  });
});
