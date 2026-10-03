async function request(path, options) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })

  const data = await response.json().catch(() => ({}))

  if (!response.ok) {
    throw new Error(data.detail || `Request failed (${response.status})`)
  }

  return data
}

export const api = {
  health: () => request('/health'),

  ingest: (force) =>
    request('/ingest', {
      method: 'POST',
      body: JSON.stringify({ force }),
    }),

  retrieve: (question, topK) =>
    request('/retrieve', {
      method: 'POST',
      body: JSON.stringify({ question, top_k: topK }),
    }),

  ask: (question, topK) =>
    request('/ask', {
      method: 'POST',
      body: JSON.stringify({ question, top_k: topK }),
    }),
}
