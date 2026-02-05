import React, { useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import '../styles/GraphVisualization.css';

const GraphVisualization = ({ graphData, loading, error }) => {
  const cyRef = useRef(null);
  const containerRef = useRef(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [layout, setLayout] = useState('cose');
  const [stats, setStats] = useState(null);

  useEffect(() => {
    if (!graphData || !containerRef.current) {
      console.log('Early return - missing graphData or container');
      return;
    }

    console.log('Creating graph with data:', graphData);

    if (!graphData.entities || !Array.isArray(graphData.entities)) {
      console.error('Invalid graphData.entities:', graphData.entities);
      return;
    }

    if (!graphData.relationships || !Array.isArray(graphData.relationships)) {
      console.error('Invalid graphData.relationships:', graphData.relationships);
      return;
    }

    // Destroy previous instance BEFORE creating new one
    if (cyRef.current) {
      console.log('Destroying previous cytoscape instance');
      try {
        cyRef.current.destroy();
      } catch (e) {
        console.warn('Error destroying previous instance:', e);
      }
      cyRef.current = null;
    }

    // Create nodes and build a Set of valid node IDs
    const nodes = graphData.entities.map(entity => ({
      data: {
        id: String(entity.id || entity.name),
        label: entity.name || 'Unknown',
        type: entity.type || 'default',
        properties: entity.properties || {},
        mentions: entity.mentions || 1
      }
    }));

    // Create a Set of all valid node IDs for quick lookup
    const validNodeIds = new Set(nodes.map(node => node.data.id));
    
    console.log('Valid node IDs:', Array.from(validNodeIds));

    // Filter edges to only include those with valid source AND target
    const edges = graphData.relationships
      .map((rel, idx) => {
        const source = String(rel.source || rel.from || '');
        const target = String(rel.target || rel.to || '');
        
        return {
          data: {
            id: `edge-${idx}`,
            source: source,
            target: target,
            label: rel.type || rel.relationship || 'related',
            type: rel.type || rel.relationship || 'related'
          },
          valid: validNodeIds.has(source) && validNodeIds.has(target)
        };
      })
      .filter(edge => {
        if (!edge.valid) {
          console.warn(`Skipping invalid edge: ${edge.data.source} -> ${edge.data.target}`);
          return false;
        }
        return true;
      })
      .map(edge => ({ data: edge.data })); // Remove the 'valid' property

    console.log('Processed nodes:', nodes.length);
    console.log('Processed edges (after validation):', edges.length);
    console.log('Sample node:', nodes[0]);
    if (edges.length > 0) {
      console.log('Sample edge:', edges[0]);
    }

    // Check container dimensions
    const rect = containerRef.current.getBoundingClientRect();
    console.log('Container dimensions:', { 
      width: rect.width, 
      height: rect.height,
      top: rect.top,
      left: rect.left
    });

    if (rect.width === 0 || rect.height === 0) {
      console.error('Container has zero dimensions! Cannot render graph.');
      return;
    }

    try {
      const cy = cytoscape({
        container: containerRef.current,
        elements: [...nodes, ...edges],
        style: [
          {
            selector: 'node',
            style: {
              'background-color': (ele) => getNodeColor(ele.data('type')),
              'label': 'data(label)',
              'width': (ele) => Math.min(20 + (ele.data('mentions') || 1) * 5, 60),
              'height': (ele) => Math.min(20 + (ele.data('mentions') || 1) * 5, 60),
              'font-size': '12px',
              'text-valign': 'center',
              'text-halign': 'center',
              'color': '#333',
              'text-outline-color': '#fff',
              'text-outline-width': 2,
              'border-width': 2,
              'border-color': '#666'
            }
          },
          {
            selector: 'node:selected',
            style: {
              'border-width': 4,
              'border-color': '#4CAF50',
              'background-color': '#81C784'
            }
          },
          {
            selector: 'edge',
            style: {
              'width': 2,
              'line-color': '#999',
              'target-arrow-color': '#999',
              'target-arrow-shape': 'triangle',
              'curve-style': 'bezier',
              'label': 'data(label)',
              'font-size': '10px',
              'text-rotation': 'autorotate',
              'text-margin-y': -10,
              'color': '#666',
              'text-outline-color': '#fff',
              'text-outline-width': 1
            }
          },
          {
            selector: 'edge:selected',
            style: {
              'line-color': '#4CAF50',
              'target-arrow-color': '#4CAF50',
              'width': 3
            }
          }
        ],
        layout: {
          name: layout,
          animate: false,  // Disable animation to prevent React StrictMode issues
          padding: 50
        }
      });

      console.log('Cytoscape instance created successfully');

      cy.on('tap', 'node', (evt) => {
        const node = evt.target;
        setSelectedNode({
          id: node.data('id'),
          label: node.data('label'),
          type: node.data('type'),
          properties: node.data('properties'),
          mentions: node.data('mentions')
        });
      });

      cy.on('tap', (evt) => {
        if (evt.target === cy) {
          setSelectedNode(null);
        }
      });

      cy.on('zoom', () => {
        setZoomLevel(cy.zoom());
      });

      cyRef.current = cy;

      setStats({
        nodes: nodes.length,
        edges: edges.length,
        entityTypes: getEntityTypeCounts(graphData.entities || []),
        relationshipTypes: getRelationshipTypeCounts(graphData.relationships || [])
      });

      // Fit the graph to the viewport after a short delay
      setTimeout(() => {
        if (cyRef.current) {
          cyRef.current.fit(50);
        }
      }, 100);

    } catch (err) {
      console.error('Error creating cytoscape graph:', err);
    }

    return () => {
      console.log('Cleanup: destroying cytoscape instance');
      if (cyRef.current) {
        try {
          cyRef.current.destroy();
        } catch (e) {
          console.warn('Error during cleanup:', e);
        }
        cyRef.current = null;
      }
    };
  }, [graphData, layout]);

  const getNodeColor = (type) => {
    const colors = {
      'PERSON': '#FF6B6B',
      'ORGANIZATION': '#4ECDC4',
      'LOCATION': '#45B7D1',
      'CONCEPT': '#FFA07A',
      'EVENT': '#98D8C8',
      'DATE': '#F7DC6F',
      'TECHNOLOGY': '#BB8FCE',
      'default': '#95A5A6'
    };
    return colors[type?.toUpperCase()] || colors.default;
  };

  const getEntityTypeCounts = (entities) => {
    return entities.reduce((acc, entity) => {
      const type = entity.type || 'Unknown';
      acc[type] = (acc[type] || 0) + 1;
      return acc;
    }, {});
  };

  const getRelationshipTypeCounts = (relationships) => {
    return relationships.reduce((acc, rel) => {
      const type = rel.type || rel.relationship || 'related';
      acc[type] = (acc[type] || 0) + 1;
      return acc;
    }, {});
  };

  const handleZoom = (delta) => {
    if (cyRef.current) {
      const currentZoom = cyRef.current.zoom();
      cyRef.current.zoom({
        level: currentZoom + delta,
        renderedPosition: { x: containerRef.current.offsetWidth / 2, y: containerRef.current.offsetHeight / 2 }
      });
    }
  };

  const changeLayout = (layoutName) => {
    setLayout(layoutName);
    if (cyRef.current) {
      cyRef.current.layout({
        name: layoutName,
        animate: true,
        animationDuration: 500
      }).run();
    }
  };

  const fitGraph = () => {
    if (cyRef.current) {
      cyRef.current.fit(50);
    }
  };

  const exportGraph = (format) => {
    if (!cyRef.current) return;

    if (format === 'png') {
      const png = cyRef.current.png({ scale: 2 });
      const link = document.createElement('a');
      link.download = 'knowledge-graph.png';
      link.href = png;
      link.click();
    } else if (format === 'json') {
      const data = JSON.stringify(graphData, null, 2);
      const blob = new Blob([data], { type: 'application/json' });
      const link = document.createElement('a');
      link.download = 'knowledge-graph.json';
      link.href = URL.createObjectURL(blob);
      link.click();
    }
  };

  if (loading) {
    return (
      <div className="graph-visualization-container">
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '100%',
          fontSize: '1.2rem',
          color: '#666'
        }}>
          Loading graph...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="graph-visualization-container">
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '100%',
          fontSize: '1.2rem',
          color: '#d32f2f',
          flexDirection: 'column',
          gap: '1rem'
        }}>
          <div>Error loading graph</div>
          <div style={{ fontSize: '0.9rem' }}>{error}</div>
        </div>
      </div>
    );
  }

  if (!graphData || !graphData.entities || graphData.entities.length === 0) {
    return (
      <div className="graph-visualization-container">
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '100%',
          fontSize: '1.2rem',
          color: '#666'
        }}>
          No graph data available
        </div>
      </div>
    );
  }

  return (
    <div className="graph-visualization-container" style={{ minHeight: '600px', height: '100vh' }}>
      <div className="graph-controls">
        <div className="control-group">
          <h3>Layout</h3>
          <div className="button-group">
            <button 
              className={`control-btn ${layout === 'cose' ? 'active' : ''}`}
              onClick={() => changeLayout('cose')}
              title="Force-directed layout"
            >
              Force
            </button>
            <button 
              className={`control-btn ${layout === 'circle' ? 'active' : ''}`}
              onClick={() => changeLayout('circle')}
              title="Circular layout"
            >
              Circle
            </button>
            <button 
              className={`control-btn ${layout === 'grid' ? 'active' : ''}`}
              onClick={() => changeLayout('grid')}
              title="Grid layout"
            >
              Grid
            </button>
            <button 
              className={`control-btn ${layout === 'breadthfirst' ? 'active' : ''}`}
              onClick={() => changeLayout('breadthfirst')}
              title="Hierarchical layout"
            >
              Tree
            </button>
          </div>
        </div>

        <div className="control-group">
          <h3>Zoom</h3>
          <div className="button-group">
            <button className="control-btn" onClick={() => handleZoom(0.2)}>➕</button>
            <span className="zoom-level">{(zoomLevel * 100).toFixed(0)}%</span>
            <button className="control-btn" onClick={() => handleZoom(-0.2)}>➖</button>
            <button className="control-btn" onClick={fitGraph}>⊡ Fit</button>
          </div>
        </div>

        <div className="control-group">
          <h3>Export</h3>
          <div className="button-group">
            <button className="control-btn" onClick={() => exportGraph('png')}>
              🖼️ PNG
            </button>
            <button className="control-btn" onClick={() => exportGraph('json')}>
              📄 JSON
            </button>
          </div>
        </div>
      </div>

      <div className="graph-content" style={{ position: 'relative', flex: 1, minHeight: '400px' }}>
        <div 
          ref={containerRef} 
          className="graph-canvas"
          style={{ 
            width: '100%', 
            height: '100%', 
            minHeight: '400px',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0
          }}
        ></div>

        {selectedNode && (
          <div className="node-details-panel">
            <div className="panel-header">
              <h3>Node Details</h3>
              <button 
                className="close-btn" 
                onClick={() => setSelectedNode(null)}
              >
                ✕
              </button>
            </div>
            <div className="panel-content">
              <div className="detail-row">
                <span className="detail-label">Name:</span>
                <span className="detail-value">{selectedNode.label}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Type:</span>
                <span className="detail-badge" style={{ backgroundColor: getNodeColor(selectedNode.type) }}>
                  {selectedNode.type}
                </span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Mentions:</span>
                <span className="detail-value">{selectedNode.mentions}</span>
              </div>
              {selectedNode.properties && Object.keys(selectedNode.properties).length > 0 && (
                <div className="detail-section">
                  <h4>Properties:</h4>
                  {Object.entries(selectedNode.properties).map(([key, value]) => (
                    <div key={key} className="property-row">
                      <span className="property-key">{key}:</span>
                      <span className="property-value">{String(value)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {stats && (
          <div className="legend-panel">
            <h3>Legend</h3>
            <div className="legend-section">
              <h4>Entity Types ({stats.nodes})</h4>
              {Object.entries(stats.entityTypes).map(([type, count]) => (
                <div key={type} className="legend-item">
                  <span 
                    className="legend-color" 
                    style={{ backgroundColor: getNodeColor(type) }}
                  ></span>
                  <span className="legend-label">{type}</span>
                  <span className="legend-count">{count}</span>
                </div>
              ))}
            </div>
            <div className="legend-section">
              <h4>Relationships ({stats.edges})</h4>
              {Object.entries(stats.relationshipTypes).map(([type, count]) => (
                <div key={type} className="legend-item">
                  <span className="legend-arrow">→</span>
                  <span className="legend-label">{type}</span>
                  <span className="legend-count">{count}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default GraphVisualization;