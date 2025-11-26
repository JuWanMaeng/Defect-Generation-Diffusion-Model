import React, { useState } from 'react';

const ImageGenerator = ({ onImageGenerated }) => {
  const [prompt, setPrompt] = useState('');
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const generateImage = async () => {
    if (!prompt) return;

    setLoading(true);
    setError(null);
    setImage(null);

    try {
      const response = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate image');
      }

      const data = await response.json();
      setImage(data.image_url);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="apple-card" style={{ maxWidth: '700px', margin: '40px auto' }}>
      <h1 className="apple-title">Imagine.</h1>
      <p className="apple-subtitle">Create stunning images with just a few words.</p>

      <div style={{ display: 'flex', gap: '12px', marginBottom: '40px' }}>
        <input
          className="apple-input"
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="A futuristic city with neon lights..."
          onKeyDown={(e) => e.key === 'Enter' && generateImage()}
        />
        <button
          className="apple-button"
          onClick={generateImage}
          disabled={loading || !prompt}
          style={{ minWidth: '120px' }}
        >
          {loading ? 'Creating...' : 'Create'}
        </button>
      </div>

      {error && (
        <div style={{ color: 'var(--error-color)', marginBottom: '20px', textAlign: 'center' }}>
          {error}
        </div>
      )}

      <div className="image-container">
        {loading ? (
          <div className="loader">Designing your masterpiece...</div>
        ) : image ? (
          <div style={{ position: 'relative', width: '100%', height: '100%' }}>
            <img
              src={image}
              alt="Generated"
              style={{
                width: '100%',
                height: 'auto',
                display: 'block',
                animation: 'fadeIn 0.8s cubic-bezier(0.2, 0.8, 0.2, 1)'
              }}
            />
            <button
              onClick={() => onImageGenerated && onImageGenerated(image)}
              className="apple-button secondary"
              style={{
                position: 'absolute',
                bottom: '24px',
                right: '24px',
                padding: '10px 20px',
                fontSize: '15px',
                backdropFilter: 'blur(20px)',
                backgroundColor: 'rgba(255, 255, 255, 0.8)',
                color: '#000',
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
              }}
            >
              Edit / Inpaint
            </button>
          </div>
        ) : (
          <div style={{ color: 'var(--text-secondary)', textAlign: 'center' }}>
            <span style={{ fontSize: '48px', display: 'block', marginBottom: '16px', opacity: 0.2 }}>🎨</span>
            Your imagination awaits.
          </div>
        )}
      </div>

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: scale(0.98); }
          to { opacity: 1; transform: scale(1); }
        }
      `}</style>
    </div>
  );
};

export default ImageGenerator;
