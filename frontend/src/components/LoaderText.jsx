import { useEffect, useState } from "react";

/**
 * Render a looping `. . . loading` animation using lightweight text updates.
 *
 * @returns {JSX.Element} Animated loading label.
 */
export default function LoaderText() {
  const [frame, setFrame] = useState(0);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      setFrame((previousFrame) => (previousFrame + 1) % 4);
    }, 280);

    return () => {
      window.clearInterval(intervalId);
    };
  }, []);

  const dots = [".", ". .", ". . .", ". . . ."][frame];

  return <span>{dots} loading</span>;
}
