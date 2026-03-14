import { computeOverallResult, formatResultSummary } from "../lib/files";

/**
 * Render the officer-facing result section for one review response.
 *
 * @param {object} props Component props.
 * @param {boolean} props.isLoading Whether a review request is currently active.
 * @param {object | null} props.reviewResult Latest review response.
 * @returns {JSX.Element} Result panel UI.
 */
export default function ResultPanel({ isLoading, reviewResult }) {
  const overallResult = reviewResult ? computeOverallResult(reviewResult) : "";

  return (
    <section className="result-panel card-surface">
      <div className="panel-header">
        <h2>Review Completeness</h2>
        <p>
          Multilayered completeness check output, status result in each layer, and
          summaries visible to IRCC Officers.
        </p>
      </div>

      {isLoading ? (
        <div className="result-placeholder shimmer-block" />
      ) : reviewResult ? (
        <>
          <div
            className={`result-banner ${
              overallResult === "PASS"
                ? "result-banner-pass"
                : overallResult === "FAIL"
                  ? "result-banner-fail"
                  : "result-banner-manual"
            }`}
          >
            Overall Result: {overallResult}
          </div>
          <div className="stage-grid">
            {reviewResult.stage_outcomes.map((stage) => (
              <article className="stage-card" key={stage.stage_name}>
                <span className={`stage-pill stage-pill-${stage.status}`}>{stage.status}</span>
                <h3>{stage.stage_name}</h3>
                <p>{stage.explanation}</p>
                {stage.evidence?.length ? (
                  <ul className="stage-evidence-list">
                    {stage.evidence.map((evidenceLine) => (
                      <li key={`${stage.stage_name}-${evidenceLine}`}>{evidenceLine}</li>
                    ))}
                  </ul>
                ) : null}
              </article>
            ))}
          </div>
          <div className="report-box">
            <pre>{formatResultSummary(reviewResult)}</pre>
          </div>
        </>
      ) : (
        <div className="empty-preview">No review result yet. Submit a package to see the report.</div>
      )}
    </section>
  );
}
