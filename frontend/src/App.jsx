import { useState } from "react";
import "./App.css";


function App() {

  // User's current question.
  const [question, setQuestion] = useState("");

  // API response.
  const [result, setResult] = useState(null);

  // Loading state.
  const [loading, setLoading] = useState(false);

  // Error message.
  const [error, setError] = useState("");


  // ---------------------------------------------------------
  // SEND QUERY
  // ---------------------------------------------------------

  async function handleAsk(event) {

    event.preventDefault();

    if (!question.trim()) {
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {

      const response = await fetch(
        "http://localhost:8000/api/query",
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            question: question,
            top_k: 5,
          }),
        }
      );

      if (!response.ok) {

        throw new Error(
          `API request failed: ${response.status}`
        );
      }

      const data = await response.json();

      setResult(data);

    } catch (err) {

      setError(
        err.message ||
        "Unable to query GraphRAG-X."
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


        {/* -------------------------------------------------
            QUESTION FORM
        -------------------------------------------------- */}

        <form
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

        </form>


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

          </section>

        )}

      </main>

    </div>
  );
}


export default App;