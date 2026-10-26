export interface DashboardIndicators {
    feedback?: { total: number; satisfaction_rate: number | null };
}

export function formatFeedback(metrics?: DashboardIndicators | null): string {
    const feedback = metrics?.feedback;
    if (!feedback) return 'Unavailable';
    if (feedback.total === 0 || feedback.satisfaction_rate == null) return 'Not rated';
    return `${(feedback.satisfaction_rate * 100).toFixed(1)}%`;
}

export function describeHealth(status?: string): string {
    switch (status) {
        case 'healthy': return 'Services available';
        case 'starting': return 'Models loading';
        case 'degraded': return 'Reduced capabilities';
        case 'unhealthy': return 'Service unavailable';
        default: return 'Health unavailable';
    }
}
