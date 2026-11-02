export type UploadStatus = 'idle' | 'uploading' | 'uploaded' | 'error';

export interface UploadStatusEntry {
    uploadStatus: UploadStatus;
}

export function isUploadCandidate(status: UploadStatus): boolean {
    return status === 'idle' || status === 'error';
}

export function getUploadCandidates<T extends UploadStatusEntry>(files: readonly T[]): T[] {
    return files.filter(file => isUploadCandidate(file.uploadStatus));
}

export function countUploadCandidates<T extends UploadStatusEntry>(files: readonly T[]): number {
    return getUploadCandidates(files).length;
}
