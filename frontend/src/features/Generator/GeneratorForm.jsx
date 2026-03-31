import React, { useState } from 'react';
import { Send, Copy, Sparkles, RefreshCw, AlertCircle } from 'lucide-react';

const GeneratorForm = ({ onJDGenerated }) => {
    const [formData, setFormData] = useState({
        jobTitle: '',
        department: '',
        experience: 'Mid-level',
        location: '',
        responsibilities: '',
        skills: '',
        tone: 'Professional'
    });

    const [generatedPrompt, setGeneratedPrompt] = useState('');
    const [isGenerating, setIsGenerating] = useState(false);
    const [isJDGenerating, setIsJDGenerating] = useState(false);
    const [error, setError] = useState(null);

    // LLM calls are proxied through the FastAPI backend at /generate-jd
    const BACKEND_URL = "http://localhost:8000";

    const handleInputChange = (e) => { //event e
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
    };

    const generatePrompt = () => {
        setIsGenerating(true);
        setError(null);

        // Standard structured prompt for the user to review
        const prompt = `Act as an expert HR Recruiter. Generate a detailed, professional job description based on the following details:

JOB TITLE: ${formData.jobTitle}
DEPARTMENT: ${formData.department}
EXPERIENCE LEVEL: ${formData.experience}
LOCATION: ${formData.location}
KEY RESPONSIBILITIES:
${formData.responsibilities}

REQUIRED SKILLS:
${formData.skills}

TONE: ${formData.tone}

Please include sections for:
1. About the Company
2. Role Summary
3. Key Responsibilities
4. Requirements & Qualifications
5. Benefits & Perks
6. How to Apply

Ensure the language is inclusive and engaging.`;

        setGeneratedPrompt(prompt);
        setIsGenerating(false);
    };

    const generateFullJD = async () => {
        setIsJDGenerating(true);
        setError(null);

        try {
            // Call the FastAPI backend which proxies the LLM request server-side
            const response = await fetch(`${BACKEND_URL}/generate-jd`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: generatedPrompt,
                    model: "google/gemini-3.1-flash-lite-preview",
                    temperature: 0.7,
                    max_tokens: 2000
                })
            });

            if (!response.ok) {
                const body = await response.json().catch(() => ({}));
                throw new Error(body.detail || `Server error: ${response.status}`);
            }

            const data = await response.json();
            const jdContent = data.content;

            onJDGenerated({
                id: Date.now(),
                title: formData.jobTitle,
                department: formData.department,
                experience: formData.experience,
                location: formData.location,
                content: jdContent,
                createdAt: new Date().toISOString()
            });
        } catch (err) {
            console.error("JD Generation Error:", err);
            const errorMessage = err.message || "An unexpected error occurred. Is the backend running?";
            setError(`Failed to generate JD: ${errorMessage}`);
        } finally {
            setIsJDGenerating(false);
        }
    };

    const copyToClipboard = () => {
        navigator.clipboard.writeText(generatedPrompt);
    };

    return (
        <div className="generator-container animate-fade-in">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                {/* Form Section */}
                <div className="card">
                    <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Sparkles size={20} color="var(--primary-color)" />
                        Job Details
                    </h3>

                    <div className="form-group">
                        <label className="form-label">Job Title</label>
                        <input
                            type="text"
                            name="jobTitle"
                            className="form-input"
                            placeholder="e.g. Senior Frontend Engineer"
                            value={formData.jobTitle}
                            onChange={handleInputChange}
                        />
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                        <div className="form-group">
                            <label className="form-label">Department</label>
                            <input
                                type="text"
                                name="department"
                                className="form-input"
                                placeholder="e.g. Engineering"
                                value={formData.department}
                                onChange={handleInputChange}
                            />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Experience</label>
                            <select
                                name="experience"
                                className="form-input"
                                value={formData.experience}
                                onChange={handleInputChange}
                            >
                                <option>Entry-level</option>
                                <option>Mid-level</option>
                                <option>Senior-level</option>
                                <option>Management</option>
                                <option>Executive</option>
                            </select>
                        </div>
                    </div>

                    <div className="form-group">
                        <label className="form-label">Location</label>
                        <input
                            type="text"
                            name="location"
                            className="form-input"
                            placeholder="e.g. Remote / New York, NY"
                            value={formData.location}
                            onChange={handleInputChange}
                        />
                    </div>

                    <div className="form-group">
                        <label className="form-label">Key Responsibilities</label>
                        <textarea
                            name="responsibilities"
                            className="form-textarea"
                            rows="4"
                            placeholder="List the main tasks and goals..."
                            value={formData.responsibilities}
                            onChange={handleInputChange}
                        ></textarea>
                    </div>

                    <div className="form-group">
                        <label className="form-label">Required Skills</label>
                        <input
                            type="text"
                            name="skills"
                            className="form-input"
                            placeholder="React, Node.js, TypeScript..."
                            value={formData.skills}
                            onChange={handleInputChange}
                        />
                    </div>

                    <button className="btn btn-primary" style={{ width: '100%' }} onClick={generatePrompt} disabled={isGenerating}>
                        {isGenerating ? <RefreshCw className="animate-spin" size={20} /> : <Send size={18} />}
                        {isGenerating ? 'Assembling Prompt...' : 'Generate AI Prompt'}
                    </button>
                </div>

                {/* Preview Section */}
                <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <Copy size={20} color="var(--accent-color)" />
                            Prompt Preview
                        </h3>
                        {generatedPrompt && (
                            <div style={{ display: 'flex', gap: '0.5rem' }}>
                                <button className="btn" style={{ padding: '0.5rem', background: 'rgba(255,255,255,0.05)' }} onClick={copyToClipboard} title="Copy Prompt">
                                    <Copy size={16} />
                                </button>
                            </div>
                        )}
                    </div>

                    <div style={{
                        flexGrow: 1,
                        background: 'rgba(0,0,0,0.2)',
                        borderRadius: '8px',
                        padding: '1rem',
                        fontFamily: 'monospace',
                        fontSize: '0.875rem',
                        color: '#cbd5e1',
                        whiteSpace: 'pre-wrap',
                        overflowY: 'auto',
                        border: '1px solid var(--border-color)',
                        marginBottom: '1.5rem'
                    }}>
                        {generatedPrompt || 'Your structured prompt will appear here after you fill out the form and click generate.'}
                    </div>

                    {error && (
                        <div style={{
                            background: 'rgba(239, 68, 68, 0.1)',
                            border: '1px solid #ef4444',
                            color: '#ef4444',
                            padding: '1rem',
                            borderRadius: '8px',
                            marginBottom: '1.5rem',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.5rem',
                            fontSize: '0.875rem'
                        }}>
                            <AlertCircle size={16} />
                            {error}
                        </div>
                    )}

                    {generatedPrompt && (
                        <button
                            className="btn btn-accent"
                            style={{ width: '100%', marginBottom: '1.5rem' }}
                            onClick={generateFullJD}
                            disabled={isJDGenerating}
                        >
                            {isJDGenerating ? <RefreshCw className="animate-spin" size={20} /> : <Sparkles size={18} />}
                            {isJDGenerating ? 'AI is Writing...' : 'Generate Full JD with AI'}
                        </button>
                    )}

                    <div style={{ marginTop: 'auto', fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                        <p>💡 Tip: Use the button above to have our AI generate the complete JD text for you automatically.</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default GeneratorForm;
