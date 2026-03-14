import { useEffect, useMemo, useState } from "react";

import FileColumn from "./components/FileColumn";
import LoaderText from "./components/LoaderText";
import ResultPanel from "./components/ResultPanel";
import { buildReviewPayload, createObjectUrlMap, revokeObjectUrlMap } from "./lib/files";
import { submitReview } from "./lib/reviewApi";

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
  const [applicationName, setApplicationName] = useState("visitor visa");
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [selectedFileId, setSelectedFileId] = useState(null);
  const [reviewResult, setReviewResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);

  const objectUrlMap = useMemo(() => createObjectUrlMap(selectedFiles), [selectedFiles]);

  useEffect(() => {
    return () => {
      revokeObjectUrlMap(objectUrlMap);
    };
  }, [objectUrlMap]);

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
          <p className="eyebrow">AI-Assisted Government Application Verification</p>
          <h1>Truthy Review Console</h1>
          <p className="hero-text">
            Upload an application package, inspect the file bundle, and send it to the FastAPI
            review endpoint from a single officer-facing screen.
          </p>
        </div>
        <div className="palette-strip" aria-hidden="true">
          <span className="swatch swatch-deep" />
          <span className="swatch swatch-crimson" />
          <span className="swatch swatch-amber" />
          <span className="swatch swatch-cream" />
        </div>
      </section>

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
            {isLoading ? <LoaderText /> : "Run Review"}
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
