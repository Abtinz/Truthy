import { afterEach, describe, expect, it, vi } from "vitest";

import { submitIndexRequest } from "./reviewApi";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("frontend API helpers", () => {
  it("submits direct indexing requests to the indexer service", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({
        status: "indexed",
        records_upserted: 4,
        logs: ["upsert_guidelines_done count=4"],
      }),
    });

    const response = await submitIndexRequest({
      source_value: "https://example.com/policy",
      index_name: "operational-guidelines-instructions",
      ingestion_mode: "crawling",
      source_title: "Example policy",
    });

    console.log("=== FRONTEND INDEX REQUEST RESPONSE ===");
    console.log(response);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe("http://localhost:8001/index");
    expect(response.status).toBe("indexed");
    expect(response.records_upserted).toBe(4);
  });

  it("throws a readable error when the indexer request fails", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      text: async () => "Index request failed.",
    });

    await expect(
      submitIndexRequest({
        source_value: "/workspace/services/data/forms/IMM5483.pdf",
        index_name: "document-checklist-pdf",
        ingestion_mode: "local_pdf",
        source_title: "Checklist",
      }),
    ).rejects.toThrow("Index request failed.");
  });
});
