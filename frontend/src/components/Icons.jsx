/**
 * Render a thick outlined acceptance icon for uploaded files.
 *
 * @returns {JSX.Element} Acceptance icon.
 */
export function CheckCircle() {
  return (
    <svg aria-hidden="true" fill="none" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2.6" />
      <path
        d="M7.5 12.3l2.8 2.9 6.2-6.5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2.8"
      />
    </svg>
  );
}
