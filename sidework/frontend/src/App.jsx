import React, { useState } from 'react';
import ImageGenerator from './components/ImageGenerator';
import InpaintingEditor from './components/InpaintingEditor';
import './App.css';

function App() {
  const [generatedImage, setGeneratedImage] = useState(null);

  return (
    <div className="App">
      {generatedImage ? (
        <InpaintingEditor
          initialImage={generatedImage}
          onBack={() => setGeneratedImage(null)}
        />
      ) : (
        <ImageGenerator onImageGenerated={setGeneratedImage} />
      )}
    </div>
  );
}

export default App;