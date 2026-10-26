import assert from 'node:assert/strict';
import test from 'node:test';
import { describeHealth, formatFeedback } from '../../src/pages/Dashboard/indicators.ts';

for (const [name, input, expected] of [
    ['missing metrics are unavailable', undefined, 'Unavailable'],
    ['legacy accuracy does not substitute for feedback', { accuracy: 100 }, 'Unavailable'],
    ['zero votes are not rated', { feedback: { total: 0, satisfaction_rate: null } }, 'Not rated'],
    ['zero positive votes remain zero percent', { feedback: { total: 2, satisfaction_rate: 0 } }, '0.0%'],
    ['rated feedback uses the measured fraction', { feedback: { total: 4, satisfaction_rate: 0.75 } }, '75.0%'],
]) {
    test(name, () => assert.equal(formatFeedback(input), expected));
}

for (const [status, expected] of [
    ['healthy', 'Services available'],
    ['starting', 'Models loading'],
    ['degraded', 'Reduced capabilities'],
    ['unhealthy', 'Service unavailable'],
    [undefined, 'Health unavailable'],
    ['unexpected', 'Health unavailable'],
]) {
    test(`health status ${status} is reported honestly`, () => {
        assert.equal(describeHealth(status), expected);
    });
}
