import { useEffect, useMemo, useState } from "react";

import FileColumn from "./components/FileColumn";
import LoaderText from "./components/LoaderText";
import ResultPanel from "./components/ResultPanel";
import { buildReviewPayload, createObjectUrlMap, revokeObjectUrlMap } from "./lib/files";
import { submitIndexRequest, submitReview } from "./lib/reviewApi";

/**
 * Resolve the active frontend page from the URL hash.
 *
 * @returns {"review" | "indexer"} Active page key.
 */
function getPageFromHash() {
  return window.location.hash === "#/indexer" ? "indexer" : "review";
}

/**
 * Render the single-page Truthy review frontend.
 *
 * The page keeps the workflow on one screen: application metadata, uploaded
 * files, file preview, and officer-facing results. The API call goes directly
 * to the FastAPI gateway service.
 *
 * @returns {JSX.Element} Main frontend application layout.
 */
export default function App() {
  const [activePage, setActivePage] = useState(getPageFromHash);
  const [applicationName, setApplicationName] = useState("visitor visa");
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [selectedFileId, setSelectedFileId] = useState(null);
  const [reviewResult, setReviewResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [indexSourceValue, setIndexSourceValue] = useState("");
  const [indexName, setIndexName] = useState("operational-guidelines-instructions");
  const [indexMode, setIndexMode] = useState("crawling");
  const [indexTitle, setIndexTitle] = useState("");
  const [indexResult, setIndexResult] = useState(null);
  const [isIndexing, setIsIndexing] = useState(false);
  const [indexErrorMessage, setIndexErrorMessage] = useState("");

  const objectUrlMap = useMemo(() => createObjectUrlMap(selectedFiles), [selectedFiles]);

  useEffect(() => {
    return () => {
      revokeObjectUrlMap(objectUrlMap);
    };
  }, [objectUrlMap]);

  useEffect(() => {
    /**
     * Sync the rendered page with browser hash changes.
     *
     * @returns {void}
     */
    function handleHashChange() {
      setActivePage(getPageFromHash());
    }

    window.addEventListener("hashchange", handleHashChange);
    return () => {
      window.removeEventListener("hashchange", handleHashChange);
    };
  }, []);

  useEffect(() => {
    setIndexName(
      indexMode === "crawling"
        ? "operational-guidelines-instructions"
        : "document-checklist-pdf",
    );
  }, [indexMode]);

  /**
   * Normalize the upload input and preserve a stable display identifier.
   *
   * @param {Event} event Upload input change event.
   * @returns {void}
   */
  function handleFileSelection(event) {
    const files = Array.from(event.target.files || []).map((file, index) => ({
      id: `${file.name}-${file.size}-${index}`,
      file,
    }));
    setSelectedFiles(files);
    setSelectedFileId(files[0]?.id ?? null);
    setReviewResult(null);
    setErrorMessage("");
  }

  /**
   * Submit the uploaded package to the FastAPI review endpoint.
   *
   * @returns {Promise<void>} Resolves when the review request completes.
   */
  async function handleSubmit() {
    setIsLoading(true);
    setErrorMessage("");

    try {
      const payload = await buildReviewPayload(applicationName, selectedFiles);
      const result = await submitReview(payload);
      setReviewResult(result);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Review request failed.");
      setReviewResult(null);
    } finally {
      setIsLoading(false);
    }
  }

  /**
   * Submit one direct source-indexing request to the indexer service.
   *
   * @returns {Promise<void>} Resolves when the indexing request completes.
   */
  async function handleIndexSubmit() {
    setIsIndexing(true);
    setIndexErrorMessage("");

    try {
      const result = await submitIndexRequest({
        source_value: indexSourceValue,
        index_name: indexName,
        ingestion_mode: indexMode,
        source_title: indexTitle,
      });
      setIndexResult(result);
    } catch (error) {
      setIndexErrorMessage(error instanceof Error ? error.message : "Index request failed.");
      setIndexResult(null);
    } finally {
      setIsIndexing(false);
    }
  }

  const selectedPreviewFile = selectedFiles.find((entry) => entry.id === selectedFileId) ?? null;

  /**
   * Open the preview modal for the currently selected file.
   *
   * @returns {void}
   */
  function openPreviewModal() {
    if (selectedPreviewFile) {
      setIsPreviewOpen(true);
    }
  }

  /**
   * Close the preview modal.
   *
   * @returns {void}
   */
  function closePreviewModal() {
    setIsPreviewOpen(false);
  }

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">AI-powered Application Completeness Check</p>
          <h1>Truthy B2G Portal</h1>
          <p className="hero-text">
            Upload an application, inspect the package, and have it inspected for the
            R10 Completeness Check in front of an IRCC Officer.
          </p>
        </div>
        <div className="hero-wordmark-card" aria-label="Immigration, Refugees and Citizenship Canada wordmark">
          <div className="hero-flagmark" aria-hidden="true">
            <span className="flag-bar" />
            <span className="flag-leaf">✦</span>
            <span className="flag-bar" />
          </div>
          <div className="hero-wordmark-text">
            <span>Immigration, Refugees</span>
            <span>and Citizenship Canada</span>
          </div>
        </div>
      </section>

      <nav className="page-nav" aria-label="Primary">
        <a
          className={`page-nav-link ${activePage === "review" ? "page-nav-link-active" : ""}`}
          href="#/review"
        >
          Review Console
        </a>
        <a
          className={`page-nav-link ${activePage === "indexer" ? "page-nav-link-active" : ""}`}
          href="#/indexer"
        >
          Indexer Console
        </a>
      </nav>

      {activePage === "review" ? (
        <section className="workspace-grid">
          <aside className="control-panel card-surface">
            <label className="field-label" htmlFor="applicationName">
              Application Name
            </label>
            <input
              id="applicationName"
              className="text-input"
              value={applicationName}
              onChange={(event) => setApplicationName(event.target.value)}
              placeholder="visitor visa"
            />

            <label className="upload-zone" htmlFor="fileUpload">
              <span className="upload-title">Upload Application Files</span>
              <span className="upload-subtitle">
                Add PDFs, images, or text-supporting files for review.
              </span>
            </label>
            <input
              id="fileUpload"
              className="hidden-input"
              type="file"
              multiple
              onChange={handleFileSelection}
            />

            <button
              className="submit-button"
              disabled={isLoading || selectedFiles.length === 0 || !applicationName.trim()}
              onClick={handleSubmit}
              type="button"
            >
              {isLoading ? <LoaderText /> : "Check Completeness"}
            </button>

            {errorMessage ? <p className="error-text">{errorMessage}</p> : null}

            <FileColumn
              files={selectedFiles}
              selectedFileId={selectedFileId}
              onSelectFile={setSelectedFileId}
              onOpenPreview={openPreviewModal}
            />
          </aside>

          <ResultPanel isLoading={isLoading} reviewResult={reviewResult} />
        </section>
      ) : (
        <section className="indexer-grid">
          <section className="indexer-panel card-surface">
            <div className="panel-header">
              <h2>Indexer Console</h2>
              <p>Submit one policy URL or one local PDF path for direct indexing.</p>
            </div>

            <label className="field-label" htmlFor="indexMode">
              Input Mode
            </label>
            <select
              id="indexMode"
              className="text-input"
              value={indexMode}
              onChange={(event) => setIndexMode(event.target.value)}
            >
              <option value="crawling">crawling</option>
              <option value="local_pdf">local pdf</option>
            </select>

            <label className="field-label" htmlFor="indexName">
              Index Name
            </label>
            <input
              id="indexName"
              className="text-input"
              value={indexName}
              onChange={(event) => setIndexName(event.target.value)}
              placeholder="operational-guidelines-instructions"
            />

            <label className="field-label" htmlFor="indexSourceValue">
              {indexMode === "crawling" ? "Source Link" : "Local PDF Path"}
            </label>
            <input
              id="indexSourceValue"
              className="text-input"
              value={indexSourceValue}
              onChange={(event) => setIndexSourceValue(event.target.value)}
              placeholder={
                indexMode === "crawling"
                  ? "https://www.canada.ca/..."
                  : "/workspace/services/data/forms/IMM5483.pdf"
              }
            />

            <label className="field-label" htmlFor="indexTitle">
              Source Title
            </label>
            <input
              id="indexTitle"
              className="text-input"
              value={indexTitle}
              onChange={(event) => setIndexTitle(event.target.value)}
              placeholder="Study permit application assessment"
            />

            <button
              className="submit-button"
              disabled={isIndexing || !indexSourceValue.trim() || !indexName.trim()}
              onClick={handleIndexSubmit}
              type="button"
            >
              {isIndexing ? <LoaderText /> : "Run Indexer"}
            </button>

            {indexErrorMessage ? <p className="error-text">{indexErrorMessage}</p> : null}
          </section>

          <section className="indexer-log-panel card-surface">
            <div className="panel-header">
              <h2>Indexer Logs</h2>
              <p>Returned runtime logs and indexing status for the submitted source.</p>
            </div>

            {isIndexing ? (
              <div className="result-placeholder shimmer-block" />
            ) : indexResult ? (
              <>
                <div className="index-result-meta">
                  <div><strong>Status:</strong> {indexResult.status}</div>
                  <div><strong>Index:</strong> {indexResult.index_name}</div>
                  <div><strong>Source Kind:</strong> {indexResult.source_kind}</div>
                  <div><strong>Modified Date:</strong> {indexResult.modified_date || "n/a"}</div>
                  <div><strong>Upserts:</strong> {indexResult.records_upserted}</div>
                </div>
                <div className="report-box">
                  <pre>{(indexResult.logs || []).join("\n")}</pre>
                </div>
              </>
            ) : (
              <div className="empty-preview">
                No indexing run yet. Submit a source to see cache decisions and upsert logs.
              </div>
            )}
          </section>
        </section>
      )}

      {isPreviewOpen && selectedPreviewFile ? (
        <div className="preview-modal-shell" onClick={closePreviewModal} role="presentation">
          <section
            className="preview-modal"
            onClick={(event) => event.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-label="File preview"
          >
            <div className="preview-modal-header">
              <div>
                <h2>File Preview</h2>
                <p>{selectedPreviewFile.file.name}</p>
              </div>
              <button className="preview-close-button" onClick={closePreviewModal} type="button">
                Close
              </button>
            </div>

            <div className="preview-frame">
              <div className="preview-meta">
                <span>{selectedPreviewFile.file.name}</span>
                <span>{selectedPreviewFile.file.type || "application/octet-stream"}</span>
              </div>
              {selectedPreviewFile.file.type.startsWith("image/") ? (
                <img
                  alt={selectedPreviewFile.file.name}
                  className="image-preview"
                  src={objectUrlMap[selectedPreviewFile.id]}
                />
              ) : selectedPreviewFile.file.type === "application/pdf" ? (
                <iframe
                  className="document-preview"
                  src={objectUrlMap[selectedPreviewFile.id]}
                  title={selectedPreviewFile.file.name}
                />
              ) : (
                <div className="empty-preview">
                  Preview is available for PDF and image files. This file will still be submitted.
                </div>
              )}
            </div>
          </section>
        </div>
      ) : null}
    </main>
  );
}
