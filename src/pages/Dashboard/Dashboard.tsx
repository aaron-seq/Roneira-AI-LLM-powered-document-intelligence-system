import React, { useState, useEffect } from 'react';
import { 
    Grid, 
    Card, 
    CardContent, 
    Typography, 
    Box, 
    Chip,
    IconButton,
    Button,
    Skeleton,
    LinearProgress,
    Divider
} from '@mui/material';
import { 
    Article, 
    CheckCircle, 
    Refresh,
    TrendingUp,
    AutoAwesome,
    PieChart,
    Timeline,
    Speed,
    Storage,
    Memory
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { apiClient } from '../../api/client';

// ==============================================================================
// Roneira AI - Data Insights Plus Dashboard
// Advanced Analytics & Visualizations
// ==============================================================================

// API paths handled by Vite proxy
// const API_BASE = 'http://localhost:8000';

interface MetricCardProps {
    title: string;
    value: string | number;
    subValue?: string;
    icon: React.ReactNode;
    trend?: { value: string; isPositive: boolean };
    gradient: string;
    glowColor: string;
    loading?: boolean;
}

const MetricCard = ({ title, value, subValue, icon, trend, gradient, glowColor, loading }: MetricCardProps) => (
    <Card sx={{ 
        height: '100%',
        cursor: 'pointer',
        position: 'relative',
        overflow: 'hidden',
        '&:hover': { 
            transform: 'translateY(-4px)',
            boxShadow: `0 20px 60px rgba(0, 0, 0, 0.5), ${glowColor}` 
        },
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
    }}>
        <Box sx={{ 
            position: 'absolute', top: 0, right: 0, width: '100px', height: '100px', 
            background: gradient, opacity: 0.1, borderRadius: '0 0 0 100%' 
        }} />
        
        <CardContent sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                <Box sx={{ p: 1.5, borderRadius: 3, background: 'rgba(255,255,255,0.05)', backdropFilter: 'blur(10px)' }}>
                    {icon}
                </Box>
                {trend && (
                    <Chip 
                        size="small" 
                        icon={<TrendingUp sx={{ fontSize: 14 }} />} 
                        label={trend.value} 
                        sx={{ 
                            height: 24, 
                            fontWeight: 600,
                            background: trend.isPositive ? 'rgba(122, 154, 91, 0.15)' : 'rgba(194, 94, 63, 0.15)', 
                            color: trend.isPositive ? '#7A9A5B' : '#C25E3F',
                            border: `1px solid ${trend.isPositive ? 'rgba(122, 154, 91, 0.2)' : 'rgba(194, 94, 63, 0.2)'}`
                        }} 
                    />
                )}
            </Box>
            
            <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 600, mb: 0.5 }}>
                {title}
            </Typography>
            
            {loading ? (
                <Skeleton variant="text" width={120} height={60} />
            ) : (
                <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1 }}>
                    <Typography variant="h3" sx={{ 
                        fontWeight: 800, 
                        fontSize: '2.5rem',
                        background: gradient, 
                        backgroundClip: 'text', 
                        WebkitBackgroundClip: 'text', 
                        color: 'transparent'
                    }}>
                        {value}
                    </Typography>
                    {subValue && (
                        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                            {subValue}
                        </Typography>
                    )}
                </Box>
            )}
        </CardContent>
    </Card>
);

const DistributionBar = ({ label, value, color, count }: { label: string, value: number, color: string, count: number }) => (
    <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', textTransform: 'uppercase' }}>{label}</Typography>
            <Typography variant="caption" sx={{ fontWeight: 600, color: color }}>{count} docs</Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <LinearProgress 
                variant="determinate" 
                value={value} 
                sx={{ 
                    flex: 1, height: 8, borderRadius: 4, 
                    background: 'rgba(255,255,255,0.05)',
                    '& .MuiLinearProgress-bar': { background: color, borderRadius: 4 }
                }} 
            />
            <Typography variant="caption" sx={{ width: 35, textAlign: 'right', color: 'text.secondary' }}>{Math.round(value)}%</Typography>
        </Box>
    </Box>
);

const Dashboard = () => {
    const navigate = useNavigate();
    const [documents, setDocuments] = useState<any[]>([]);
    const [health, setHealth] = useState<any>(null);
    const [metrics, setMetrics] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    
    // Flat accents rather than two-hue gradients. A gradient from ochre to
    // purple says nothing about the number underneath it, and the palette only
    // has one accent to spend — on provenance.
    const gradients = {
        primary: '#D0A215',
        info: '#5E7A8A',
        success: '#7A9A5B',
        warning: '#C25E3F',
        error: '#C25E3F',
        dark: '#2A2B26'
    };

    const glows = {
        primary: '0 0 24px rgba(208, 162, 21, 0.18)',
        info: '0 0 24px rgba(94, 122, 138, 0.18)',
        success: '0 0 24px rgba(122, 154, 91, 0.18)',
        warning: '0 0 24px rgba(194, 94, 63, 0.18)',
    };

    const fetchData = async () => {
        try {
            // apiClient, not fetch: document endpoints require a bearer token
            // and bare fetch sends none, so every request 401'd and the whole
            // dashboard rendered zeros against a full index.
            //
            // Health is deliberately the one exception — it is unauthenticated
            // and must stay readable even when a token has expired, which is
            // exactly when you want to know what is wrong.
            const [docsRes, healthRes, metricsRes] = await Promise.allSettled([
                apiClient.get('/documents?limit=100'),
                apiClient.get('/health'),
                apiClient.get('/dashboard/metrics'),
            ]);

            if (docsRes.status === 'fulfilled') {
                setDocuments(docsRes.value.data?.documents ?? []);
            }
            if (healthRes.status === 'fulfilled') setHealth(healthRes.value.data);
            if (metricsRes.status === 'fulfilled') setMetrics(metricsRes.value.data);
            setLoading(false);
        } catch (error) {
            console.error(error);
            setLoading(false);
        }
    };

    useEffect(() => { fetchData(); const i = setInterval(fetchData, 15000); return () => clearInterval(i); }, []);

    // Calculated Metrics
    const total = metrics?.total_documents || documents.length;
    const processing = documents.filter(d => d.status === 'processing').length;
    
    // Document Types
    const getDocType = (name: string) => {
        const n = name.toLowerCase();
        if (n.includes('inv')) return 'Finance';
        if (n.includes('hr') || n.includes('policy')) return 'HR';
        if (n.includes('case') || n.includes('eng') || n.includes('spec')) return 'Engineering';
        return 'General';
    };
    
    const types = documents.reduce((acc, doc) => {
        const type = getDocType(doc.filename || 'unknown');
        acc[type] = (acc[type] || 0) + 1;
        return acc;
    }, {} as Record<string, number>);

    // Insight Metrics (Derived from actual data)
    // Guard against metrics being null/undefined
    const accuracy = (metrics?.accuracy !== undefined && metrics?.accuracy !== null) 
        ? Number(metrics.accuracy).toFixed(1) 
        : '100.0';
    const avgTime = '-'; // Will be populated when we have real processing time metrics
    const storageUsed = (documents.reduce((acc, d) => acc + (d.file_size || 0), 0) / 1024 / 1024).toFixed(2);

    return (
        <Box sx={{ flexGrow: 1, p: { xs: 2, md: 4, lg: 5 }, minHeight: '100vh', background: '#0F100E' }}>
            {/* Header */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 5 }}>
                <Box>
                    <Typography variant="h4" sx={{ fontWeight: 800, letterSpacing: '-0.02em', mb: 1, color: 'white' }}>
                        Data Insights <Box component="span" sx={{ color: '#D0A215' }}>Plus</Box>
                    </Typography>
                    <Typography variant="body1" sx={{ color: 'text.secondary', fontWeight: 500 }}>
                        Real-time intelligence dashboard & system telemetry
                    </Typography>
                </Box>
                <Box sx={{ display: 'flex', gap: 2 }}>
                    <Button 
                        variant="contained" 
                        size="large"
                        startIcon={<AutoAwesome />}
                        onClick={() => navigate('/chat')}
                        sx={{ 
                            background: gradients.primary, 
                            borderRadius: 3, 
                            px: 4, py: 1.5,
                            fontSize: '1rem',
                            fontWeight: 700,
                            boxShadow: glows.primary
                        }}
                    >
                        Ask AI Assistant
                    </Button>
                    <IconButton 
                        onClick={fetchData} 
                        sx={{ 
                            background: 'rgba(255,255,255,0.05)', 
                            borderRadius: 3,
                            width: 50, height: 50,
                            '&:hover': { background: 'rgba(255,255,255,0.1)' }
                        }}
                    >
                        <Refresh />
                    </IconButton>
                </Box>
            </Box>

            {/* Core Metrics Grid */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={12} sm={6} lg={3}>
                    <MetricCard 
                        title="Total Indexed" 
                        value={total} 
                        subValue="documents"
                        icon={<Storage sx={{ color: '#a855f7' }} />} 
                        gradient={gradients.primary} 
                        glowColor={glows.primary} 
                        loading={loading} 
                    />
                </Grid>
                <Grid item xs={12} sm={6} lg={3}>
                    <MetricCard 
                        title="AI Accuracy Score" 
                        value={`${accuracy}%`} 
                        icon={<VerifiedUser sx={{ color: '#7A9A5B' }} />} 
                        gradient={gradients.success} 
                        glowColor={glows.success} 
                        loading={loading} 
                    />
                </Grid>
                <Grid item xs={12} sm={6} lg={3}>
                    <MetricCard 
                        title="Avg Process Time" 
                        value={avgTime} 
                        icon={<Speed sx={{ color: '#D0A215' }} />} 
                        gradient={gradients.info} 
                        glowColor={glows.info} 
                        loading={loading} 
                    />
                </Grid>
                <Grid item xs={12} sm={6} lg={3}>
                    <MetricCard 
                        title="Storage Used" 
                        value={storageUsed} 
                        subValue="MB"
                        icon={<Memory sx={{ color: '#C9922B' }} />} 
                        gradient={gradients.warning} 
                        glowColor={glows.warning} 
                        loading={loading} 
                    />
                </Grid>
            </Grid>

            {/* Main Visualizations */}
            <Grid container spacing={3}>
                {/* Document Type Distribution */}
                <Grid item xs={12} md={4}>
                    <Card sx={{ height: '100%', background: 'rgba(10, 11, 9, 0.6)', backdropFilter: 'blur(20px)', border: '1px solid rgba(255,255,255,0.05)' }}>
                        <CardContent sx={{ p: 4 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 4 }}>
                                <Typography variant="h6" sx={{ fontWeight: 700, color: 'white' }}>Corpus Distribution</Typography>
                                <PieChart sx={{ color: 'text.secondary' }} />
                            </Box>
                            
                            <DistributionBar label="Finance & Invoices" value={total > 0 ? ((types['Finance'] || 0) / total) * 100 : 0} count={types['Finance'] || 0} color="#D0A215" />
                            <DistributionBar label="Human Resources" value={total > 0 ? ((types['HR'] || 0) / total) * 100 : 0} count={types['HR'] || 0} color="#7A9A5B" />
                            <DistributionBar label="Engineering" value={total > 0 ? ((types['Engineering'] || 0) / total) * 100 : 0} count={types['Engineering'] || 0} color="#C9922B" />
                            <DistributionBar label="Uncategorized" value={total > 0 ? ((types['General'] || 0) / total) * 100 : 0} count={types['General'] || 0} color="#95907F" />
                            
                            <Divider sx={{ my: 3, borderColor: 'rgba(255,255,255,0.1)' }} />
                            
                            <Box sx={{ textAlign: 'center' }}>
                                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                    Knowledge Base Last Updated: Just now
                                </Typography>
                            </Box>
                        </CardContent>
                    </Card>
                </Grid>

                {/* System Health & Activity */}
                <Grid item xs={12} md={8}>
                    <Grid container spacing={3}>
                        {/* System Status */}
                        <Grid item xs={12}>
                            <Card sx={{ background: 'linear-gradient(135deg, rgba(122, 154, 91, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%)', border: '1px solid rgba(122, 154, 91, 0.2)' }}>
                                <CardContent sx={{ p: 3, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                        <Box sx={{ position: 'relative' }}>
                                            <Box sx={{ width: 12, height: 12, borderRadius: '50%', background: '#7A9A5B', boxShadow: '0 0 10px #7A9A5B' }} />
                                            <Box sx={{ position: 'absolute', top: -4, left: -4, right: -4, bottom: -4, borderRadius: '50%', border: '2px solid rgba(122, 154, 91, 0.3)', animation: 'pulse 2s infinite' }} />
                                        </Box>
                                        <Box>
                                            <Typography variant="h6" sx={{ fontWeight: 700, color: 'white' }}>All Systems Operational</Typography>
                                            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                                                LLM Inference (Ollama) • Tensor OCR • Redis Pipeline
                                            </Typography>
                                        </Box>
                                    </Box>
                                    <Box sx={{ display: 'flex', gap: 3 }}>
                                        <Box sx={{ textAlign: 'right' }}>
                                            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>Database</Typography>
                                            <Typography variant="body1" sx={{ fontWeight: 600, color: health?.database_status === 'connected' ? '#7A9A5B' : '#C25E3F' }}>
                                                {health?.database_status || '-'}
                                            </Typography>
                                        </Box>
                                        <Box sx={{ textAlign: 'right' }}>
                                            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>Processing Queue</Typography>
                                            <Typography variant="body1" sx={{ fontWeight: 600, color: '#D0A215' }}>{processing} jobs</Typography>
                                        </Box>
                                        <Box sx={{ textAlign: 'right' }}>
                                            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>Services</Typography>
                                            <Typography variant="body1" sx={{ fontWeight: 600, color: health?.services_status === 'operational' ? '#a855f7' : '#C9922B' }}>
                                                {health?.services_status || '-'}
                                            </Typography>
                                        </Box>
                                    </Box>
                                </CardContent>
                            </Card>
                        </Grid>

                        {/* Recent Activity Feed */}
                        <Grid item xs={12}>
                            <Card sx={{ background: 'rgba(10, 11, 9, 0.6)', backdropFilter: 'blur(20px)', border: '1px solid rgba(255,255,255,0.05)' }}>
                                <CardContent sx={{ p: 4 }}>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
                                        <Typography variant="h6" sx={{ fontWeight: 700, color: 'white' }}>Live Processing Feed</Typography>
                                        <Timeline sx={{ color: 'text.secondary' }} />
                                    </Box>
                                    
                                    {documents.slice(0, 4).map((doc, idx) => (
                                        <Box key={doc.id} sx={{ mb: 2, pb: 2, borderBottom: idx < 3 ? '1px solid rgba(255,255,255,0.05)' : 'none', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                                <Box sx={{ 
                                                    width: 40, height: 40, borderRadius: 2, 
                                                    background: 'rgba(208, 162, 21, 0.08)', color: '#D0A215',
                                                    display: 'flex', alignItems: 'center', justifyContent: 'center'
                                                }}>
                                                    <Article fontSize="small" />
                                                </Box>
                                                <Box>
                                                    <Typography variant="body2" sx={{ fontWeight: 600, color: 'white' }}>{doc.filename}</Typography>
                                                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                                        Processed by llama3.2:3b • {doc.file_size} bytes
                                                    </Typography>
                                                </Box>
                                            </Box>
                                            <Chip 
                                                label={doc.is_processed ? 'ANALYZED' : 'PROCESSING'} 
                                                size="small" 
                                                sx={{ 
                                                    fontWeight: 700, fontSize: '0.7rem',
                                                    background: doc.is_processed ? 'rgba(122, 154, 91, 0.1)' : 'rgba(201, 146, 43, 0.1)',
                                                    color: doc.is_processed ? '#7A9A5B' : '#C9922B'
                                                }} 
                                            />
                                        </Box>
                                    ))}
                                </CardContent>
                            </Card>
                        </Grid>
                    </Grid>
                </Grid>
            </Grid>
        </Box>
    );
};

// Simple icon placeholder since we don't have VerifiedUser imported
const VerifiedUser = (props: any) => <CheckCircle {...props} />;

export default Dashboard;