import { SearchResult, ExportResult } from '../types/api';

export const transformResultsForExport = (
  results: SearchResult[],
  query: string
): ExportResult[] => {
  return results.map((result, index) => {
    // Extract similarity score
    const similarityScore = result.similarity_score || result.combined_score || 0;
    const similarityPercentage = `${(similarityScore * 100).toFixed(1)}%`;
    
    // Determine relation type based on similarity score
    let relationType = 'analog';
    if (similarityScore >= 0.95) {
      relationType = 'duplicate';
    } else if (similarityScore >= 0.8) {
      relationType = 'high_similarity';
    } else if (similarityScore >= 0.6) {
      relationType = 'analog';
    } else {
      relationType = 'low_similarity';
    }
    
    // Determine search method
    const searchMethod = result.search_method || 
      (result.similarity_score ? 'semantic' : 
       result.combined_score ? 'hybrid' : 
       result.fuzzy_score ? 'fuzzy' : 'unknown');
    
    // Generate suggested category (placeholder logic)
    const suggestedCategory = extractCategory(result.document);
    
    // Generate final decision based on similarity
    let finalDecision = 'review_required';
    if (similarityScore >= 0.9) {
      finalDecision = 'approved_analog';
    } else if (similarityScore >= 0.7) {
      finalDecision = 'potential_analog';
    } else {
      finalDecision = 'requires_manual_review';
    }
    
    // Generate comment
    const comment = generateComment(result, similarityScore, searchMethod);
    
    return {
      Raw_Name: query,
      Cleaned_Name: cleanName(query),
      Lemmatized_Name: lemmatizeName(query),
      Normalized_Name: normalizeName(query),
      Candidate_Name: result.document,
      Similarity_Score: similarityPercentage,
      Relation_Type: relationType,
      Suggested_Category: suggestedCategory,
      Final_Decision: finalDecision,
      Comment: comment
    };
  });
};

const cleanName = (name: string): string => {
  // Basic cleaning - remove extra spaces, normalize case
  return name.trim().replace(/\s+/g, ' ');
};

const lemmatizeName = (name: string): string => {
  // Placeholder for lemmatization - in real implementation this would be done by backend
  return cleanName(name).toLowerCase();
};

const normalizeName = (name: string): string => {
  // Placeholder for normalization - in real implementation this would be done by backend
  return lemmatizeName(name).replace(/[^\w\s]/g, '');
};

const extractCategory = (productName: string): string => {
  // Simple category extraction based on keywords
  const name = productName.toLowerCase();
  
  if (name.includes('болт') || name.includes('винт')) {
    return 'Крепежные изделия - Болты и винты';
  } else if (name.includes('гайка')) {
    return 'Крепежные изделия - Гайки';
  } else if (name.includes('шайба')) {
    return 'Крепежные изделия - Шайбы';
  } else if (name.includes('труба')) {
    return 'Трубы и трубопроводная арматура';
  } else if (name.includes('кабель') || name.includes('провод')) {
    return 'Электротехническая продукция';
  } else {
    return 'Общие материалы';
  }
};

const generateComment = (
  result: SearchResult, 
  similarityScore: number, 
  searchMethod: string
): string => {
  const comments: string[] = [];
  
  // Add similarity-based comment
  if (similarityScore >= 0.9) {
    comments.push('Высокая степень схожести');
  } else if (similarityScore >= 0.7) {
    comments.push('Хорошая степень схожести');
  } else if (similarityScore >= 0.5) {
    comments.push('Умеренная степень схожести');
  } else {
    comments.push('Низкая степень схожести');
  }
  
  // Add search method comment
  comments.push(`Найдено методом: ${searchMethod}`);
  
  // Add additional metrics if available
  if (result.cosine_score) {
    comments.push(`Косинусное расстояние: ${(result.cosine_score * 100).toFixed(1)}%`);
  }
  
  if (result.fuzzy_score) {
    comments.push(`Нечеткое соответствие: ${result.fuzzy_score}`);
  }
  
  return comments.join('; ');
};

export const downloadExcelFile = async (
  results: SearchResult[],
  query: string,
  filename?: string
): Promise<void> => {
  try {
    // Transform results to export format
    const exportData = transformResultsForExport(results, query);
    
    // Prepare the request payload
    const exportRequest = {
      [query]: results
    };
    
    const response = await fetch('http://localhost:8000/search/export-results', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(exportRequest),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Export failed');
    }
    
    // Get the blob from response
    const blob = await response.blob();
    
    // Create download link
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename || `analog_search_results_${new Date().toISOString().split('T')[0]}.xlsx`;
    
    // Trigger download
    document.body.appendChild(link);
    link.click();
    
    // Cleanup
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    
  } catch (error) {
    console.error('Export failed:', error);
    throw error;
  }
};

export const generateCSVContent = (results: SearchResult[], query: string): string => {
  const exportData = transformResultsForExport(results, query);
  
  // CSV headers
  const headers = [
    'Raw_Name',
    'Cleaned_Name', 
    'Lemmatized_Name',
    'Normalized_Name',
    'Candidate_Name',
    'Similarity_Score',
    'Relation_Type',
    'Suggested_Category',
    'Final_Decision',
    'Comment'
  ];
  
  // Convert to CSV format
  const csvRows = [
    headers.join(','),
    ...exportData.map(row => 
      headers.map(header => {
        const value = row[header as keyof ExportResult];
        // Escape commas and quotes in CSV
        return `"${String(value).replace(/"/g, '""')}"`;
      }).join(',')
    )
  ];
  
  return csvRows.join('\n');
};

export const downloadCSVFile = (results: SearchResult[], query: string, filename?: string): void => {
  const csvContent = generateCSVContent(results, query);
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename || `analog_search_results_${new Date().toISOString().split('T')[0]}.csv`;
  
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};
