import React, { useState } from 'react';
import URLInput from './components/URLInput';
import GraphVisualization from './components/GraphVisualisation';
import LoadingSpinner from './components/Loadingspinner';
import ErrorDisplay from './components/Error';
import './styles/App.css';

function App() {
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');

  const handleURLSubmit = async (url) => {
    setLoading(true);
    setError(null);
    setProgress(0);
    setGraphData(null);
    setCurrentStep('Starting pipeline...');

    try {
      setProgress(10);
      setCurrentStep('Sending request to backend...');

      const response = await fetch('http://localhost:8000/api/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: url }),
      });

      setProgress(30);
      setCurrentStep('Backend processing article...');

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      setProgress(70);
      setCurrentStep('Receiving knowledge graph...');

      const data = await response.json();

      if (!data.entities || !data.relationships) {
        throw new Error('Invalid response format from backend');
      }

      setProgress(90);
      setCurrentStep('Rendering graph...');

      await new Promise(resolve => setTimeout(resolve, 300));

      setGraphData(data);
      setProgress(100);
      setCurrentStep('Complete!');
      
      setTimeout(() => {
        setLoading(false);
      }, 500);

    } catch (err) {
      console.error('Error:', err);
      setError(err.message);
      setLoading(false);
      setProgress(0);
      setCurrentStep('');
    }
  };

  const handleRetry = () => {
    setError(null);
    setGraphData(null);
    setProgress(0);
    setCurrentStep('');
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>📊 Knowledge Graph Generator</h1>
        <p>Extract entities and relationships from any article</p>
      </header>

      <main className="app-main">
        {!graphData && !loading && !error && (
          <div className="input-section">
            <URLInput onSubmit={handleURLSubmit} loading={loading} />
          </div>
        )}

        {loading && (
          <div className="loading-section">
            <LoadingSpinner message={currentStep} progress={progress} />
          </div>
        )}

        {error && (
          <div className="error-section">
            <ErrorDisplay error={error} onRetry={handleRetry} />
          </div>
        )}

        {graphData && !loading && !error && (
          <div className="graph-section">
            <GraphVisualization 
              graphData={graphData} 
              loading={loading}
              error={error}
            />
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>Powered by GPT-4 and Cytoscape.js</p>
      </footer>
    </div>
  );
}

export default App;