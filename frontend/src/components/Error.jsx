import React from 'react';
import '../styles/error.css';

const ErrorDisplay = ({ error, onRetry }) => {
  const formatError = (errorMessage) => {
    const errorMap = {
      'Failed to fetch': 'Network error. Please check your connection.',
      'HTTP error! status: 404': 'Article not found.',
      'HTTP error! status: 500': 'Server error. Please try again later.',
      'HTTP error! status: 403': 'Access to this URL is forbidden.',
      'Invalid URL': 'Please enter a valid URL.',
    };

    for (const [key, value] of Object.entries(errorMap)) {
      if (errorMessage.includes(key)) {
        return value;
      }
    }

    return errorMessage;
  };

  const getErrorIcon = (errorMessage) => {
    if (errorMessage.includes('Network') || errorMessage.includes('fetch')) {
      return 'Network Error';
    }
    if (errorMessage.includes('not found') || errorMessage.includes('404')) {
      return 'Not Found';
    }
    if (errorMessage.includes('forbidden') || errorMessage.includes('403')) {
      return 'Access Denied';
    }
    if (errorMessage.includes('server') || errorMessage.includes('500')) {
      return 'Server Error';
    }
    return 'Error';
  };

  return (
    <div className="error-display-container">
      <div className="error-card">
        <div className="error-icon-large">
          {getErrorIcon(error)}
        </div>

        <h2 className="error-title">Something went wrong</h2>
        
        <p className="error-message">
          {formatError(error)}
        </p>

        <div className="error-details">
          <details>
            <summary>Technical details</summary>
            <code className="error-code">{error}</code>
          </details>
        </div>

        {onRetry && (
          <div className="error-actions">
            <button className="retry-button" onClick={onRetry}>
              Retry
            </button>
          </div>
        )}

        <div className="error-suggestions">
          <h4>Suggestions</h4>
          <ul>
            <li>Check that the URL is correct and accessible</li>
            <li>Ensure the article is publicly available</li>
            <li>Try a different news site or article</li>
            <li>Verify your internet connection</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ErrorDisplay;
