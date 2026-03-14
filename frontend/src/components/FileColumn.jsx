import { CheckCircle } from "./Icons";

/**
 * Render the uploaded file list with selection and accepted-state markers.
 *
 * @param {object} props Component props.
 * @param {{id: string, file: File}[]} props.files Uploaded file entries.
 * @param {string | null} props.selectedFileId Currently selected file identifier.
 * @param {(fileId: string) => void} props.onSelectFile Selection callback.
 * @param {() => void} props.onOpenPreview Preview callback for the selected file.
 * @returns {JSX.Element} Uploaded file column UI.
 */
export default function FileColumn({ files, selectedFileId, onSelectFile, onOpenPreview }) {
  return (
    <div className="file-column">
      <div className="panel-header">
        <h2>Uploaded Files</h2>
        <p>{files.length} file(s) staged for review.</p>
      </div>
      <button
        className="secondary-button"
        disabled={!selectedFileId}
        onClick={onOpenPreview}
        type="button"
      >
        Open Selected Preview
      </button>
      <div className="file-list">
        {files.length === 0 ? (
          <div className="empty-file-state">Upload files to populate the review bundle.</div>
        ) : (
          files.map((entry) => (
            <button
              key={entry.id}
              className={`file-chip ${selectedFileId === entry.id ? "file-chip-active" : ""}`}
              onClick={() => onSelectFile(entry.id)}
              type="button"
            >
              <span className="file-chip-mark">
                <CheckCircle />
              </span>
              <span className="file-chip-copy">
                <strong>{entry.file.name}</strong>
                <small>{entry.file.type || "Unknown type"}</small>
              </span>
            </button>
          ))
        )}
      </div>
    </div>
  );
}
