import assert from 'node:assert/strict';
import test from 'node:test';
import { messagesThroughAssistant } from '../../src/pages/AIChat/conversation.ts';

const history = [
    { id: 'welcome', role: 'assistant', content: 'Welcome' },
    { id: 'q1', role: 'user', content: 'Q1' },
    { id: 'a1', role: 'assistant', content: 'A1' },
    { id: 'q2', role: 'user', content: 'Q2' },
    { id: 'a2', role: 'assistant', content: 'A2' },
];

test('regenerating an earlier answer branches away later visible turns', () => {
    assert.deepEqual(
        messagesThroughAssistant(history, 'a1'),
        history.slice(0, 3),
    );
});

test('regenerating the last answer keeps the visible conversation', () => {
    assert.deepEqual(
        messagesThroughAssistant(history, 'a2'),
        history,
    );
});

test('welcome messages and non-adjacent answers are not regeneration targets', () => {
    assert.equal(messagesThroughAssistant(history, 'welcome'), null);
    assert.equal(
        messagesThroughAssistant([
            { id: 'q1', role: 'user', content: 'Q1' },
            { id: 'a1', role: 'assistant', content: 'A1' },
            { id: 'a2', role: 'assistant', content: 'A2' },
        ], 'a2'),
        null,
    );
});
