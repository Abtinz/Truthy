import {
  computeOverallResult,
  formatResultSummary,
} from "./files";

describe("frontend file helpers", () => {
  it("computes PASS only when all stages pass", () => {
    const result = computeOverallResult({
      stage_outcomes: [
        { stage_name: "Document Presence", status: "passed" },
        { stage_name: "Form Completion", status: "passed" },
        { stage_name: "Content Sufficiency", status: "passed" },
      ],
    });

    console.log("=== FRONTEND OVERALL RESULT PASS ===");
    console.log(result);

    expect(result).toBe("PASS");
  });

  it("computes FAIL when any stage fails", () => {
    const result = computeOverallResult({
      stage_outcomes: [
        { stage_name: "Document Presence", status: "passed" },
        { stage_name: "Form Completion", status: "passed" },
        { stage_name: "Content Sufficiency", status: "failed" },
      ],
    });

    console.log("=== FRONTEND OVERALL RESULT FAIL ===");
    console.log(result);

    expect(result).toBe("FAIL");
  });

  it("formats the result summary for the report panel", () => {
    const summary = formatResultSummary({
      application_name: "visitor visa",
      stage_outcomes: [
        {
          stage_name: "Document Presence",
          status: "passed",
          explanation: "Required document categories were identified.",
        },
      ],
      final_report_text: "Officer review is still required before any final decision.",
    });

    console.log("=== FRONTEND RESULT SUMMARY ===");
    console.log(summary);

    expect(summary).toContain("Application Name: visitor visa");
    expect(summary).toContain("Overall Result: PASS");
    expect(summary).toContain("Officer review is still required");
  });
});
