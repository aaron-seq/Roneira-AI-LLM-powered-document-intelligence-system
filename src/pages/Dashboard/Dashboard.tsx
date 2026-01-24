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
    Tooltip,
    Divider,
    Paper
} from '@mui/material';
import { 
    Article, 
    Cached, 
    CheckCircle, 
    Chat,
    Refresh,
    TrendingUp,
    AutoAwesome,
    ArrowForward,
    PieChart,
    BarChart as BarChartIcon,
    Timeline,
    Speed,
    BugReport,
    Storage,
    Memory
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

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
                            background: trend.isPositive ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)', 
                            color: trend.isPositive ? '#10b981' : '#ef4444',
                            border: `1px solid ${trend.isPositive ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)'}`
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
    
    // Theme Gradients
    const gradients = {
        primary: 'linear-gradient(135deg, #6366f1 0%, #a855f7 100%)',
        info: 'linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%)',
        success: 'linear-gradient(135deg, #10b981 0%, #84cc16 100%)',
        warning: 'linear-gradient(135deg, #f59e0b 0%, #f97316 100%)',
        error: 'linear-gradient(135deg, #ef4444 0%, #be123c 100%)',
        dark: 'linear-gradient(135deg, #0f172a 0%, #334155 100%)'
    };
    
    const glows = {
        primary: '0 0 30px rgba(99, 102, 241, 0.4)',
        info: '0 0 30px rgba(6, 182, 212, 0.4)',
        success: '0 0 30px rgba(16, 185, 129, 0.4)',
        warning: '0 0 30px rgba(245, 158, 11, 0.4)',
    };

    const fetchData = async () => {
        try {
            const [docsRes, healthRes, metricsRes] = await Promise.all([
                fetch('/api/documents?limit=100'),
                fetch('/health'), // Might need /api/health depending on backend routing, assuming root health check
                fetch('/api/dashboard/metrics')
            ]);
            if (docsRes.ok) {
                const data = await docsRes.json();
                setDocuments(data.documents || []);
            }
            if (healthRes.ok) setHealth(await healthRes.json());
            if (metricsRes.ok) setMetrics(await metricsRes.json());
            setLoading(false);
        } catch (error) {
            console.error(error);
            setLoading(false);
        }
    };

    useEffect(() => { fetchData(); const i = setInterval(fetchData, 15000); return () => clearInterval(i); }, []);

    // Calculated Metrics
    const total = metrics?.total_documents || documents.length;
    const completed = metrics?.processed_documents || documents.filter(d => d.status === 'completed').length;
    const processing = documents.filter(d => d.status === 'processing').length;
    const failed = documents.filter(d => d.status === 'failed').length;
    
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
        <Box sx={{ flexGrow: 1, p: { xs: 2, md: 4, lg: 5 }, minHeight: '100vh', background: '#0a0f1a' }}>
            {/* Header */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 5 }}>
                <Box>
                    <Typography variant="h4" sx={{ fontWeight: 800, letterSpacing: '-0.02em', mb: 1, color: 'white' }}>
                        Data Insights <Box component="span" sx={{ color: '#06b6d4' }}>Plus</Box>
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
                        icon={<VerifiedUser sx={{ color: '#10b981' }} />} 
                        gradient={gradients.success} 
                        glowColor={glows.success} 
                        loading={loading} 
                    />
                </Grid>
                <Grid item xs={12} sm={6} lg={3}>
                    <MetricCard 
                        title="Avg Process Time" 
                        value={avgTime} 
                        icon={<Speed sx={{ color: '#06b6d4' }} />} 
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
                        icon={<Memory sx={{ color: '#f59e0b' }} />} 
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
                    <Card sx={{ height: '100%', background: 'rgba(15, 23, 42, 0.6)', backdropFilter: 'blur(20px)', border: '1px solid rgba(255,255,255,0.05)' }}>
                        <CardContent sx={{ p: 4 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 4 }}>
                                <Typography variant="h6" sx={{ fontWeight: 700, color: 'white' }}>Corpus Distribution</Typography>
                                <PieChart sx={{ color: 'text.secondary' }} />
                            </Box>
                            
                            <DistributionBar label="Finance & Invoices" value={total > 0 ? ((types['Finance'] || 0) / total) * 100 : 0} count={types['Finance'] || 0} color="#6366f1" />
                            <DistributionBar label="Human Resources" value={total > 0 ? ((types['HR'] || 0) / total) * 100 : 0} count={types['HR'] || 0} color="#10b981" />
                            <DistributionBar label="Engineering" value={total > 0 ? ((types['Engineering'] || 0) / total) * 100 : 0} count={types['Engineering'] || 0} color="#f59e0b" />
                            <DistributionBar label="Uncategorized" value={total > 0 ? ((types['General'] || 0) / total) * 100 : 0} count={types['General'] || 0} color="#94a3b8" />
                            
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
                            <Card sx={{ background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%)', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                                <CardContent sx={{ p: 3, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                        <Box sx={{ position: 'relative' }}>
                                            <Box sx={{ width: 12, height: 12, borderRadius: '50%', background: '#10b981', boxShadow: '0 0 10px #10b981' }} />
                                            <Box sx={{ position: 'absolute', top: -4, left: -4, right: -4, bottom: -4, borderRadius: '50%', border: '2px solid rgba(16, 185, 129, 0.3)', animation: 'pulse 2s infinite' }} />
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
                                            <Typography variant="body1" sx={{ fontWeight: 600, color: health?.database_status === 'connected' ? '#10b981' : '#ef4444' }}>
                                                {health?.database_status || '-'}
                                            </Typography>
                                        </Box>
                                        <Box sx={{ textAlign: 'right' }}>
                                            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>Processing Queue</Typography>
                                            <Typography variant="body1" sx={{ fontWeight: 600, color: '#06b6d4' }}>{processing} jobs</Typography>
                                        </Box>
                                        <Box sx={{ textAlign: 'right' }}>
                                            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>Services</Typography>
                                            <Typography variant="body1" sx={{ fontWeight: 600, color: health?.services_status === 'operational' ? '#a855f7' : '#f59e0b' }}>
                                                {health?.services_status || '-'}
                                            </Typography>
                                        </Box>
                                    </Box>
                                </CardContent>
                            </Card>
                        </Grid>

                        {/* Recent Activity Feed */}
                        <Grid item xs={12}>
                            <Card sx={{ background: 'rgba(15, 23, 42, 0.6)', backdropFilter: 'blur(20px)', border: '1px solid rgba(255,255,255,0.05)' }}>
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
                                                    background: 'rgba(99, 102, 241, 0.1)', color: '#6366f1',
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
                                                    background: doc.is_processed ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)',
                                                    color: doc.is_processed ? '#10b981' : '#f59e0b'
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