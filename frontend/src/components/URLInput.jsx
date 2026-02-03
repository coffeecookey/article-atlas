import React, { useState } from 'react';
import '../styles/URLinput.css';

const URLInput = ({ onSubmit, loading }) => {
  const [url, setUrl] = useState('');
  const [validationError, setValidationError] = useState(null);

  const validateURL = (urlString) => {
    if (!urlString.trim()) {
      return 'Please enter a URL';
    }

    try {
      const urlObj = new URL(urlString);
      
      if (!['http:', 'https:'].includes(urlObj.protocol)) {
        return 'URL must start with http:// or https://';
      }

      if (!urlObj.hostname || urlObj.hostname.length < 3) {
        return 'Invalid URL format';
      }

      return null;
    } catch (e) {
      return 'Invalid URL format';
    }
  };

  const handleChange = (e) => {
    const newUrl = e.target.value;
    setUrl(newUrl);
    
    if (validationError) {
      setValidationError(null);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    const error = validateURL(url);
    if (error) {
      setValidationError(error);
      return;
    }

    setValidationError(null);
    onSubmit(url);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSubmit(e);
    }
  };

  return (
    <div className="url-input-container">
      <div className="url-input-card">
        <h2>Enter Article URL</h2>
        <p className="url-input-description">
          Paste any article URL to generate a knowledge graph
        </p>

        <form onSubmit={handleSubmit} className="url-input-form">
          <div className="input-wrapper">
            <input
              type="text"
              value={url}
              onChange={handleChange}
              onKeyPress={handleKeyPress}
              placeholder="https://example.com/article"
              className={`url-input ${validationError ? 'error' : ''}`}
              disabled={loading}
              autoFocus
            />
            <button 
              type="submit" 
              className="submit-button"
              disabled={loading || !url.trim()}
            >
              {loading ? 'Processing...' : 'Generate Graph'}
            </button>
          </div>

          {validationError && (
            <div className="validation-error">
              <span className="error-icon">⚠️</span>
              {validationError}
            </div>
          )}
        </form>

        <div className="example-urls">
          <p className="example-label">Try these examples:</p>
          <div className="example-buttons">
            <button 
              className="example-button"
              onClick={() => setUrl('https://www.bbc.com/news/technology')}
              disabled={loading}
            >
              BBC Tech News
            </button>
            <button 
              className="example-button"
              onClick={() => setUrl('https://en.wikipedia.org/wiki/Artificial_intelligence')}
              disabled={loading}
            >
              Wikipedia AI
            </button>
            <button 
              className="example-button"
              onClick={() => setUrl('https://www.theguardian.com/science')}
              disabled={loading}
            >
              Guardian Science
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default URLInput;