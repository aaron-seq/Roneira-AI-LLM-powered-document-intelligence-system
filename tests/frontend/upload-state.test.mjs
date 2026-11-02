import assert from 'node:assert/strict';
import test from 'node:test';
import {
    countUploadCandidates,
    getUploadCandidates,
    isUploadCandidate,
} from '../../src/pages/DocumentUpload/uploadState.ts';

test('the same retained file can retry after a transient upload failure', async () => {
    const file = { name: 'invoice.pdf' };
    const row = { file, uploadStatus: 'idle' };
    let attempts = 0;

    const attemptUpload = async () => {
        for (const candidate of getUploadCandidates([row])) {
            candidate.uploadStatus = 'uploading';
            try {
                attempts += 1;
                if (attempts === 1) throw new Error('temporary failure');
                candidate.uploadStatus = 'uploaded';
            } catch {
                candidate.uploadStatus = 'error';
            }
        }
    };

    await attemptUpload();
    assert.equal(row.uploadStatus, 'error');
    assert.equal(countUploadCandidates([row]), 1);
    assert.equal(getUploadCandidates([row])[0], row);
    assert.equal(getUploadCandidates([row])[0].file, file);

    await attemptUpload();
    assert.equal(row.uploadStatus, 'uploaded');
    assert.equal(countUploadCandidates([row]), 0);

    await attemptUpload();
    assert.equal(attempts, 2);
});

test('mixed batches count retryable rows, expose them for removal, and preserve uploaded rows', () => {
    const files = [
        { name: 'new.pdf', uploadStatus: 'idle' },
        { name: 'retry.pdf', uploadStatus: 'error' },
        { name: 'done.pdf', uploadStatus: 'uploaded' },
        { name: 'active.pdf', uploadStatus: 'uploading' },
    ];

    const candidates = getUploadCandidates(files);
    assert.deepEqual(candidates.map(file => file.name), ['new.pdf', 'retry.pdf']);
    assert.equal(countUploadCandidates(files), 2);
    assert.equal(isUploadCandidate(files[1].uploadStatus), true);
    assert.equal(isUploadCandidate(files[2].uploadStatus), false);
});
