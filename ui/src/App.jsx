import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api.js'

function StatusBar({
  storeSize,
  onRefresh,
  onIngest,
  ingesting,
  force,
  onForceChange,
}) {
  return (
    <header className="status-bar">
      <div className="status-left">
        <h1>Knowledge Base Agent</h1>
        <span
          className={
            storeSize === null
              ? 'pill pill-error'
              : storeSize > 0
                ? 'pill pill-ok'
                : 'pill pill-empty'
          }
        >
          {storeSize === null
            ? 'API unreachable'
            : storeSize > 0
              ? `${storeSize} chunks indexed`
              : 'Vector store empty'}
        </span>
      </div>

      <div className="status-actions">
        <label className="force-toggle" title="Re-chunk and re-embed everything">
          <input
            type="checkbox"
            checked={force}
            onChange={(event) => onForceChange(event.target.checked)}
          />
          Force rebuild
        </label>
        <button
          className="btn btn-secondary"
          onClick={onRefresh}
          disabled={ingesting}
        >
          Refresh
        </button>
        <button
          className="btn btn-primary"
          onClick={onIngest}
          disabled={ingesting}
        >
          {ingesting ? 'Ingesting...' : 'Ingest documents'}
        </button>
      </div>
    </header>
  )
}

function SourceCard({ source, index }) {
  const [expanded, setExpanded] = useState(index === 0)

  return (
    <div className="source-card">
      <button
        className="source-header"
        onClick={() => setExpanded((value) => !value)}
      >
        <span className="source-label">S{index + 1}</span>
        <span className="source-file">
          {source.filename} · page {source.page_number}
        </span>
        <span className="source-score">
          {source.score.toFixed(4)}
          <span
            className="score-bar"
            style={{ width: `${Math.max(2, source.score * 100)}%` }}
          />
        </span>
        <span className="source-chevron">{expanded ? '▾' : '▸'}</span>
      </button>

      {expanded && (
        <p className="source-content">{source.content}</p>
      )}
    </div>
  )
}

function Exchange({ exchange, isLast, loading }) {
  return (
    <section className="exchange">
      <div className="question">{exchange.question}</div>

      {loading && isLast ? (
        <div className="answer loading">Searching knowledge base...</div>
      ) : (
        <>
          <div className="answer">{exchange.answer}</div>

          {exchange.error && <div className="error">{exchange.error}</div>}

          {exchange.sources.length > 0 && (
            <div className="sources">
              <h3>Sources ({exchange.sources.length})</h3>
              {exchange.sources.map((source, index) => (
                <SourceCard
                  key={source.chunk_id || index}
                  source={source}
                  index={index}
                />
              ))}
            </div>
          )}
        </>
      )}
    </section>
  )
}

export default function App() {
  const [storeSize, setStoreSize] = useState(null)
  const [exchanges, setExchanges] = useState([])
  const [question, setQuestion] = useState('')
  const [pending, setPending] = useState(false)
  const [ingesting, setIngesting] = useState(false)
  const [force, setForce] = useState(false)
  const [topK, setTopK] = useState(5)
  const [notice, setNotice] = useState('')

  const bottomRef = useRef(null)

  const refresh = useCallback(async () => {
    try {
      const data = await api.health()
      setStoreSize(data.vector_store_size)
    } catch {
      setStoreSize(null)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [exchanges, pending])

  async function handleIngest() {
    setIngesting(true)
    setNotice('')

    try {
      const data = await api.ingest(force)
      await refresh()
      setNotice(`Ingest complete: ${data.chunks_stored} chunks stored.`)
    } catch (exc) {
      setNotice(`Ingest failed: ${exc.message}`)
    } finally {
      setIngesting(false)
    }
  }

  async function handleAsk(event) {
    event.preventDefault()

    const trimmed = question.trim()

    if (!trimmed || pending) {
      return
    }

    setQuestion('')
    setPending(true)
    setNotice('')

    const current = { question: trimmed, answer: '', sources: [], error: '' }
    setExchanges((previous) => [...previous, current])

    try {
      const data = await api.ask(trimmed, topK)

      setExchanges((previous) =>
        previous.map((item) =>
          item.question === trimmed && item.answer === ''
            ? {
                ...item,
                answer: data.answer,
                sources: data.sources,
              }
            : item,
        ),
      )
    } catch (exc) {
      setExchanges((previous) =>
        previous.map((item) =>
          item.question === trimmed && item.answer === ''
            ? { ...item, error: exc.message }
            : item,
        ),
      )
    } finally {
      setPending(false)
    }
  }

  return (
    <div className="app">
      <StatusBar
        storeSize={storeSize}
        onRefresh={refresh}
        onIngest={handleIngest}
        ingesting={ingesting}
        force={force}
        onForceChange={setForce}
      />

      {notice && (
        <div className="notice" role="status">
          {notice}
        </div>
      )}

      <main className="chat">
        {exchanges.length === 0 && !pending && (
          <div className="empty-state">
            <p>Ask a question about your research papers.</p>
            <p className="hint">
              Answers are grounded in the retrieved chunks and cite sources
              like [S1].
            </p>
          </div>
        )}

        {exchanges.map((exchange, index) => (
          <Exchange
            key={`${exchange.question}-${index}`}
            exchange={exchange}
            isLast={index === exchanges.length - 1}
            loading={pending && index === exchanges.length - 1}
          />
        ))}

        <div ref={bottomRef} />
      </main>

      <form className="composer" onSubmit={handleAsk}>
        <label className="topk">
          Top-K
          <select
            value={topK}
            onChange={(event) => setTopK(Number(event.target.value))}
          >
            {[1, 2, 3, 5, 8, 10].map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>

        <input
          type="text"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask about your documents..."
          disabled={pending}
          autoFocus
        />

        <button className="btn btn-primary" type="submit" disabled={pending}>
          Ask
        </button>
      </form>
    </div>
  )
}
