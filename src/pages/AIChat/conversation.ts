export function messagesThroughAssistant<T extends { id: string; role: 'user' | 'assistant' }>(
    messages: readonly T[],
    assistantId: string,
): T[] | null {
    const assistantIndex = messages.findIndex(message => message.id === assistantId);
    if (
        assistantIndex <= 0
        || messages[assistantIndex].role !== 'assistant'
        || messages[assistantIndex - 1].role !== 'user'
    ) {
        return null;
    }

    return messages.slice(0, assistantIndex + 1);
}
