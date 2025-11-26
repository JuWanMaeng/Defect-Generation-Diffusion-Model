import React, { useState, useRef } from 'react';

const InpaintingEditor = ({ initialImage, onBack }) => {
    const [image, setImage] = useState(initialImage);
    const [prompt, setPrompt] = useState('');
    const [loading, setLoading] = useState(false);
    const [bbox, setBbox] = useState(null); // {x, y, w, h}
    const [isDrawing, setIsDrawing] = useState(false);
    const [startPos, setStartPos] = useState({ x: 0, y: 0 });

    const imgRef = useRef(null);
    const containerRef = useRef(null);

    const getMousePos = (e) => {
        const rect = imgRef.current.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    };

    const handleMouseDown = (e) => {
        if (loading) return;
        const pos = getMousePos(e);
        setStartPos(pos);
        setIsDrawing(true);
        setBbox({ x: pos.x, y: pos.y, w: 0, h: 0 });
    };

    const handleMouseMove = (e) => {
        if (!isDrawing || loading) return;
        const pos = getMousePos(e);

        const w = pos.x - startPos.x;
        const h = pos.y - startPos.y;

        setBbox({
            x: w > 0 ? startPos.x : pos.x,
            y: h > 0 ? startPos.y : pos.y,
            w: Math.abs(w),
            h: Math.abs(h)
        });
    };

    const handleMouseUp = () => {
        setIsDrawing(false);
    };

    const handleInpaint = async () => {
        if (!bbox || !prompt) return;

        setLoading(true);
        try {
            const img = imgRef.current;
            const scaleX = img.naturalWidth / img.width;
            const scaleY = img.naturalHeight / img.height;

            const realBbox = [
                Math.round(bbox.x * scaleX),
                Math.round(bbox.y * scaleY),
                Math.round(bbox.w * scaleX),
                Math.round(bbox.h * scaleY)
            ];

            const response = await fetch('http://localhost:8000/inpaint', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: image,
                    prompt: prompt,
                    bbox: realBbox
                }),
            });

            if (!response.ok) throw new Error('Inpainting failed');

            const data = await response.json();
            setImage(data.image_url);
            setBbox(null);
        } catch (err) {
            console.error(err);
            alert('Inpainting failed: ' + err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="apple-card" style={{ maxWidth: '900px', margin: '40px auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
                <button
                    className="apple-button secondary"
                    onClick={onBack}
                    style={{ padding: '8px 16px', fontSize: '15px' }}
                >
                    ← Back
                </button>
                <h2 className="apple-title" style={{ fontSize: '32px', margin: 0 }}>Refine.</h2>
                <div style={{ width: '80px' }}></div>
            </div>

            <div
                ref={containerRef}
                className="image-container"
                style={{
                    display: 'inline-block',
                    marginBottom: '30px',
                    cursor: isDrawing ? 'crosshair' : 'default',
                    userSelect: 'none',
                    overflow: 'hidden',
                    width: '100%'
                }}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
            >
                <img
                    ref={imgRef}
                    src={image}
                    alt="To inpaint"
                    style={{ width: '100%', display: 'block', pointerEvents: 'none' }}
                />

                {bbox && (
                    <div style={{
                        position: 'absolute',
                        left: bbox.x,
                        top: bbox.y,
                        width: bbox.w,
                        height: bbox.h,
                        border: '2px solid #0071e3',
                        backgroundColor: 'rgba(0, 113, 227, 0.2)',
                        pointerEvents: 'none',
                        boxShadow: '0 0 0 1px rgba(255,255,255,0.5)'
                    }} />
                )}
            </div>

            <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                <input
                    className="apple-input"
                    type="text"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="What should replace the selected area?"
                    disabled={loading}
                />
                <button
                    className="apple-button"
                    onClick={handleInpaint}
                    disabled={loading || !bbox || !prompt}
                    style={{ minWidth: '140px' }}
                >
                    {loading ? 'Refining...' : 'Refine'}
                </button>
            </div>

            <p style={{ color: 'var(--text-secondary)', marginTop: '16px', fontSize: '14px', textAlign: 'center' }}>
                Select an area on the image, then describe the change you want to make.
            </p>
        </div>
    );
};

export default InpaintingEditor;
