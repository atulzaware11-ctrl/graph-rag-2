import { useState } from "react";
import "./App.css";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function App() {

  // User's current question.
  const [question, setQuestion] = useState("");

  // API response.
  const [result, setResult] = useState(null);

  // Loading state.
  const [loading, setLoading] = useState(false);

  // Error message.
  const [error, setError] = useState("");
  const [activeDocument, setActiveDocument] = useState(null);
  const [sourceFile, setSourceFile] = useState(null);
  const [uploadState, setUploadState] = useState("");
  const [uploadError, setUploadError] = useState("");
  const [graphState, setGraphState] = useState("");
  const [graphError, setGraphError] = useState("");

  function startNewDocument() {
    setActiveDocument(null);
    setQuestion("");
    setResult(null);
    setError("");
    setUploadError("");
    setUploadState("");
    setSourceFile(null);
    setGraphState("");
    setGraphError("");
  }

  async function handleUpload(event) {
    event.preventDefault();
    const file = event.target.elements.pdf.files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Choose a PDF file to upload.");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setUploadError("PDF files must be 20 MB or smaller.");
      return;
    }
    setUploadError("");
    setUploadState("Uploading...");
    const indexingTimer = setTimeout(() => setUploadState("Indexing..."), 700);
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(`${API_BASE_URL}/api/documents/upload`, {
        method: "POST",
        body: form,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const messages = {
          400: "That file is not a valid PDF.",
          413: "PDF files must be 20 MB or smaller.",
          500: "Document indexing failed. Please try again.",
        };
        throw new Error(messages[response.status] || "Unable to upload this PDF.");
      }
      setActiveDocument(data);
      setSourceFile(file);
      setResult(null);
      setQuestion("");
      setUploadState("Ready");
      setGraphState("");
      setGraphError("");
      clearTimeout(indexingTimer);
      event.target.reset();
    } catch (err) {
      clearTimeout(indexingTimer);
      setUploadError(err.message === "Failed to fetch"
        ? "GraphRAG-X is unavailable. Check that the backend is running."
        : err.message || "Unable to upload this PDF.");
      setUploadState("");
    }
  }

  async function handleBuildGraph() {
    if (!activeDocument || !sourceFile) return;

    setGraphState("Building...");
    setGraphError("");
    try {
      const form = new FormData();
      form.append("document_id", activeDocument.document_id);
      form.append("file", sourceFile);
      const response = await fetch(`${API_BASE_URL}/api/graph/build`, {
        method: "POST",
        body: form,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        if (data.error === "llm_quota_exceeded") {
          throw new Error(data.message || "Knowledge graph generation has reached the current free-model request limit.");
        }
        throw new Error(data.detail || "Knowledge graph build failed.");
      }
      setGraphState(`Ready: ${data.entities_extracted || 0} entities`);
    } catch (err) {
      setGraphState("");
      setGraphError(err.message === "Failed to fetch"
        ? "GraphRAG-X is unavailable. Check that the backend is running."
        : err.message || "Knowledge graph build failed.");
    }
  }


  // ---------------------------------------------------------
  // SEND QUERY
  // ---------------------------------------------------------

  async function handleAsk(event) {

    event.preventDefault();

    if (!question.trim()) {
      setError("Enter a question before searching.");
      return;
    }
    if (!activeDocument) {
      setError("Upload a PDF before asking a question.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {

      const response = await fetch(
        `${API_BASE_URL}/api/query`,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            document_id: activeDocument.document_id,
            question: question,
            top_k: 5,
          }),
        }
      );

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        if (data.error === "llm_quota_exceeded" && data.retrieval_available) {
          setResult(data);
          setError(data.message || "AI generation is temporarily unavailable. Retrieved evidence is shown below.");
          return;
        }

        const messages = {
          404: "This document is no longer available. Upload it again to continue.",
          409: "This document is still indexing. Try again when it is ready.",
          422: "Enter a question of up to 2,000 characters.",
        };
        throw new Error(messages[response.status] || "Unable to answer that question.");
      }

      setResult(data);

    } catch (err) {

      setError(
        err.message === "Failed to fetch"
          ? "GraphRAG-X is unavailable. Check that the backend is running."
          : err.message || "Unable to query GraphRAG-X."
      );

    } finally {

      setLoading(false);

    }
  }


  // ---------------------------------------------------------
  // RENDER
  // ---------------------------------------------------------

  return (
    <div className="app">

      <header className="header">

        <div>
          <h1>GraphRAG-X</h1>

          <p>
            Adaptive Graph + Vector Research Intelligence
          </p>
        </div>

        <div className="status">
          AI Research System
        </div>

      </header>


      <main className="container">

        <section className="hero">

          <span className="eyebrow">
            RESEARCH INTELLIGENCE
          </span>

          <h2>
            Ask questions across your technical documents.
          </h2>

          <p>
            GraphRAG-X combines vector retrieval, BM25,
            knowledge graphs and reranking to find
            evidence-backed answers.
          </p>

        </section>

        <section className="answer-card document-panel">
          <div className="card-header">
            <h3>{activeDocument ? "Active document" : "Upload a PDF"}</h3>
            {activeDocument && <button type="button" onClick={startNewDocument}>New Document</button>}
          </div>
          {!activeDocument && (
            <form onSubmit={handleUpload}>
              <input name="pdf" type="file" accept="application/pdf,.pdf" />
              <button type="submit" disabled={uploadState === "Uploading..." || uploadState === "Indexing..."}>
                {uploadState === "Uploading..." || uploadState === "Indexing..." ? uploadState : "Upload PDF"}
              </button>
            </form>
          )}
          {uploadState && <p>{uploadState}</p>}
          {uploadError && <div className="error">{uploadError}</div>}
          {activeDocument && (
            <div>
              <strong>{activeDocument.filename}</strong>
              <p>{activeDocument.pages_processed} pages · {activeDocument.chunks_processed} chunks · {activeDocument.status}</p>
              <small>Document ID: {activeDocument.document_id.slice(0, 8)}…</small>
              <div>
                <button type="button" onClick={handleBuildGraph} disabled={!sourceFile || graphState === "Building..."}>
                  {graphState === "Building..." ? "Building..." : "Build Knowledge Graph"}
                </button>
              </div>
              {graphState && <p>{graphState}</p>}
              {graphError && <div className="error">{graphError}</div>}
            </div>
          )}
        </section>


        {/* -------------------------------------------------
            QUESTION FORM
        -------------------------------------------------- */}

        {activeDocument && <form
          className="query-form"
          onSubmit={handleAsk}
        >

          <textarea
            value={question}
            onChange={(event) =>
              setQuestion(event.target.value)
            }
            placeholder="Ask a research question..."
            rows="3"
          />

          <button
            type="submit"
            disabled={loading}
          >

            {loading
              ? "Researching..."
              : "Ask GraphRAG-X"}

          </button>

        </form>}


        {/* -------------------------------------------------
            ERROR
        -------------------------------------------------- */}

        {error && (

          <div className="error">
            {error}
          </div>

        )}


        {/* -------------------------------------------------
            RESULT
        -------------------------------------------------- */}

        {result && (

          <section className="results">

            {/* ANSWER */}

            <div className="answer-card">

              <div className="card-header">

                <h3>Answer</h3>

                {result.confidence && (

                  <span
                    className={`confidence ${
                      result.confidence.label
                        .toLowerCase()
                    }`}
                  >
                    {result.confidence.label}
                    {" "}
                    {result.confidence.score}%
                  </span>

                )}

              </div>

              <div className="answer">

                {result.answer}

              </div>

            </div>


            {/* GRAPH */}

            <div className="graph-card">

              <div className="card-header">

                <h3>Knowledge Graph</h3>

                <span>
                  {result.graph_entities || 0}
                  {" "}
                  entities
                </span>

              </div>

              {result.graph_context &&
              result.graph_context.length > 0 ? (

                <div className="graph-list">

                  {result.graph_context.map(
                    (item, index) => (

                      <div
                        className="graph-item"
                        key={index}
                      >

                        <strong>
                          {item.entity}
                        </strong>

                        <span>
                          {item.entity_type}
                        </span>

                        {item.relationships &&
                        item.relationships.length > 0 && (

                          <div className="relationships">

                            {item.relationships
                              .slice(0, 5)
                              .map(
                                (
                                  relationship,
                                  relationshipIndex
                                ) => (

                                  <div
                                    key={
                                      relationshipIndex
                                    }
                                  >

                                    {item.entity}

                                    {" → "}

                                    {
                                      relationship.relationship
                                    }

                                    {" → "}

                                    {
                                      relationship.connected_entity
                                    }

                                  </div>

                                )
                              )}

                          </div>

                        )}

                      </div>

                    )
                  )}

                </div>

              ) : (

                <p className="muted">
                  No graph entities were found
                  for this question.
                </p>

              )}

            </div>


            {/* EVIDENCE */}

            <div className="evidence-card">

              <div className="card-header">

                <h3>Evidence</h3>

                <span>
                  {result.sources?.length || 0}
                  {" "}
                  sources
                </span>

              </div>


              {result.sources?.map(
                (source) => (

                  <article
                    className="source"
                    key={source.source}
                  >

                    <div className="source-meta">

                      <strong>
                        Source {source.source}
                      </strong>

                      <span>
                        Page {source.page}
                      </span>

                      <span>
                        Evidence{" "}
                        {Math.round(
                          source.evidence_score * 100
                        )}
                        %
                      </span>

                    </div>

                    <p>
                      {source.text}
                    </p>

                  </article>

                )
              )}

            </div>


            {/* SYSTEM INFORMATION */}

            <div className="system-card">

              <div>
                <span>Retrieval</span>
                <strong>
                  {result.retrieval_method}
                </strong>
              </div>

              <div>
                <span>Retrieved</span>
                <strong>
                  {result.retrieved_chunks}
                </strong>
              </div>

              <div>
                <span>Reranked</span>
                <strong>
                  {result.reranked_chunks}
                </strong>
              </div>

              <div>
                <span>Graph Entities</span>
                <strong>
                  {result.graph_entities}
                </strong>
              </div>

            </div>

            {result.timings && (
              <div className="system-card">
                <div><span>Retrieval</span><strong>{result.timings.retrieval_seconds}s</strong></div>
                <div><span>Reranking</span><strong>{result.timings.rerank_seconds}s</strong></div>
                <div><span>LLM</span><strong>{result.timings.llm_seconds}s</strong></div>
                <div><span>Total</span><strong>{result.timings.total_seconds}s</strong></div>
              </div>
            )}

          </section>

        )}

      </main>

    </div>
  );
}


export default App;
