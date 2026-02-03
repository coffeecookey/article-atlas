import React from 'react';
import '../styles/LoadingSpinner.css';

const LoadingSpinner = ({ message = 'Processing...', progress }) => {
  return (
    <div className="loading-spinner-container">
      <div className="loading-card">
        <div className="spinner">
          <div className="spinner-circle"></div>
          <div className="spinner-circle"></div>
          <div className="spinner-circle"></div>
        </div>

        <h3 className="loading-message">{message}</h3>

        {progress !== undefined && progress !== null && (
          <div className="progress-container">
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            <span className="progress-text">{progress}%</span>
          </div>
        )}

        <div className="loading-steps">
          <div className={`step ${progress >= 20 ? 'active' : ''}`}>
            <span className="step-icon">📄</span>
            <span className="step-label">Scraping</span>
          </div>
          <div className={`step ${progress >= 40 ? 'active' : ''}`}>
            <span className="step-icon">🧹</span>
            <span className="step-label">Cleaning</span>
          </div>
          <div className={`step ${progress >= 60 ? 'active' : ''}`}>
            <span className="step-icon">✂️</span>
            <span className="step-label">Chunking</span>
          </div>
          <div className={`step ${progress >= 80 ? 'active' : ''}`}>
            <span className="step-icon">🤖</span>
            <span className="step-label">Extracting</span>
          </div>
          <div className={`step ${progress >= 100 ? 'active' : ''}`}>
            <span className="step-icon">📊</span>
            <span className="step-label">Building Graph</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoadingSpinner;