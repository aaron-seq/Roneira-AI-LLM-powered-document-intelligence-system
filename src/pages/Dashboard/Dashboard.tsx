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
    LinearProgress
} from '@mui/material';
import { 
    Article, 
    CheckCircle, 
    Refresh,
    AutoAwesome,
    PieChart,
    Timeline,
    Storage,
    Memory
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { apiClient } from '../../api/client';
import { describeHealth, formatFeedback } from './indicators';

interface MetricCardProps {
    title: string;
    value: string | number;
    subValue?: string;
    icon: React.ReactNode;
    gradient: string;
    glowColor: string;
    loading?: boolean;
}

const MetricCard = ({ title, value, subValue, icon, gradient, glowColor, loading }: MetricCardProps) => (
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
    // has one accent to spend â€” on provenance.
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
            // Health is deliberately the one exception â€” it is unauthenticated
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
            setHealth(healthRes.status === 'fulfilled' ? healthRes.value.data : null);
            setMetrics(metricsRes.status === 'fulfilled' ? metricsRes.value.data : null);
            setLoading(false);
        } catch (error) {
            console.error(error);
            setLoading(false);
        }
    };

    useEffect(() => { fetchData(); const i = setInterval(fetchData, 15000); return () => clearInterval(i); }, []);

    // Calculated Metrics
    const total = metrics?.total_documents ?? documents.length;
    const processing = metrics?.documents_by_status?.processing ?? documents.filter(d => d.status === 'processing').length;
    
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

    const feedbackRating = formatFeedback(metrics);
    const storageUsed = (documents.reduce((acc, d) => acc + (d.size_bytes ?? 0), 0) / 1024 / 1024).toFixed(2);
    const healthColor = health?.status === 'healthy' ? '#7A9A5B' : '#C25E3F';

    return (
        <Box sx={{ flexGrow: 1, p: { xs: 2, md: 4, lg: 5 }, minHeight: '100vh', background: '#0F100E' }}>
            {/* Header */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 5 }}>
                <Box>
                    <Typography variant="h4" sx={{ fontWeight: 800, letterSpacing: '-0.02em', mb: 1, color: 'white' }}>
                        Document overview
                    </Typography>
                    <Typography variant="body1" sx={{ color: 'text.secondary', fontWeight: 500 }}>
                        Documents, answer feedback, and service availability
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
                        aria-label="Refresh dashboard"
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
                <Grid item xs={12} sm={6} lg={4}>
                    <MetricCard 
                        title="Documents"
                        value={total} 
                        subValue="documents"
                        icon={<Storage sx={{ color: '#a855f7' }} />} 
                        gradient={gradients.primary} 
                        glowColor={glows.primary} 
                        loading={loading} 
                    />
                </Grid>
                <Grid item xs={12} sm={6} lg={4}>
                    <MetricCard 
                        title="Positive answer feedback"
                        value={feedbackRating}
                        subValue={metrics?.feedback ? `${metrics.feedback.total} ratings` : undefined}
                        icon={<CheckCircle sx={{ color: '#7A9A5B' }} />}
                        gradient={gradients.success} 
                        glowColor={glows.success} 
                        loading={loading} 
                    />
                </Grid>
                <Grid item xs={12} sm={6} lg={4}>
                    <MetricCard 
                        title="Loaded file size"
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
                                <Typography variant="h6" sx={{ fontWeight: 700, color: 'white' }}>Filename groups</Typography>
                                <PieChart sx={{ color: 'text.secondary' }} />
                            </Box>
                            
                            <DistributionBar label="Finance & Invoices" value={documents.length > 0 ? ((types['Finance'] || 0) / documents.length) * 100 : 0} count={types['Finance'] || 0} color="#D0A215" />
                            <DistributionBar label="Human Resources" value={documents.length > 0 ? ((types['HR'] || 0) / documents.length) * 100 : 0} count={types['HR'] || 0} color="#7A9A5B" />
                            <DistributionBar label="Engineering" value={documents.length > 0 ? ((types['Engineering'] || 0) / documents.length) * 100 : 0} count={types['Engineering'] || 0} color="#C9922B" />
                            <DistributionBar label="Uncategorized" value={documents.length > 0 ? ((types['General'] || 0) / documents.length) * 100 : 0} count={types['General'] || 0} color="#95907F" />
                            
                            <Box sx={{ textAlign: 'center' }}>
                                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                    Grouped from the loaded document filenames
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
                            <Card sx={{ background: 'rgba(255,255,255,0.03)', border: `1px solid ${healthColor}` }}>
                                <CardContent sx={{ p: 3, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                        <Box sx={{ position: 'relative' }}>
                                            <Box sx={{ width: 12, height: 12, borderRadius: '50%', background: healthColor }} />
                                        </Box>
                                        <Box>
                                            <Typography variant="h6" sx={{ fontWeight: 700, color: 'white' }}>{describeHealth(health?.status)}</Typography>
                                            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                                                Document and model service availability
                                            </Typography>
                                        </Box>
                                    </Box>
                                    <Box sx={{ display: 'flex', gap: 3 }}>
                                        <Box sx={{ textAlign: 'right' }}>
                                            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>Database</Typography>
                                            <Typography variant="body1" sx={{ fontWeight: 600, color: health?.components?.database?.status === 'ok' ? '#7A9A5B' : '#C25E3F' }}>
                                                {health?.components?.database?.status ?? 'unknown'}
                                            </Typography>
                                        </Box>
                                        <Box sx={{ textAlign: 'right' }}>
                                            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>Processing</Typography>
                                            <Typography variant="body1" sx={{ fontWeight: 600, color: '#D0A215' }}>{processing} documents</Typography>
                                        </Box>
                                        <Box sx={{ textAlign: 'right' }}>
                                            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>Services</Typography>
                                            <Typography variant="body1" sx={{ fontWeight: 600, color: healthColor }}>
                                                {health?.status ?? 'unknown'}
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
                                        <Typography variant="h6" sx={{ fontWeight: 700, color: 'white' }}>Recent documents</Typography>
                                        <Timeline sx={{ color: 'text.secondary' }} />
                                    </Box>
                                    
                                    {documents.slice(0, 4).map((doc, idx) => (
                                        <Box key={doc.document_id} sx={{ mb: 2, pb: 2, borderBottom: idx < 3 ? '1px solid rgba(255,255,255,0.05)' : 'none', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
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
                                                        {doc.size_bytes} bytes
                                                    </Typography>
                                                </Box>
                                            </Box>
                                            <Chip 
                                                label={String(doc.status ?? 'unknown').toUpperCase()}
                                                size="small" 
                                                sx={{ 
                                                    fontWeight: 700, fontSize: '0.7rem',
                                                    background: doc.status === 'completed' ? 'rgba(122, 154, 91, 0.1)' : 'rgba(201, 146, 43, 0.1)',
                                                    color: doc.status === 'completed' ? '#7A9A5B' : '#C9922B'
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

export default Dashboard;
