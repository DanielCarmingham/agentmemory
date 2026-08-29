import type { EmbeddingProvider } from "../../types.js";
import { detectEmbeddingProvider, getEnvVar } from "../../config.js";
import { bootLog, logger } from "../../logger.js";
import { GeminiEmbeddingProvider } from "./gemini.js";
import { OpenAIEmbeddingProvider } from "./openai.js";
import { VoyageEmbeddingProvider } from "./voyage.js";
import { CohereEmbeddingProvider } from "./cohere.js";
import { OpenRouterEmbeddingProvider } from "./openrouter.js";
import { LocalEmbeddingProvider } from "./local.js";
import { ClipEmbeddingProvider } from "./clip.js";

export {
  GeminiEmbeddingProvider,
  OpenAIEmbeddingProvider,
  VoyageEmbeddingProvider,
  CohereEmbeddingProvider,
  OpenRouterEmbeddingProvider,
  LocalEmbeddingProvider,
  ClipEmbeddingProvider,
};

let imageEmbeddingProvider: EmbeddingProvider | null = null;

export function createImageEmbeddingProvider(): EmbeddingProvider | null {
  if (process.env["AGENTMEMORY_IMAGE_EMBEDDINGS"] !== "true") return null;
  if (imageEmbeddingProvider) return imageEmbeddingProvider;
  imageEmbeddingProvider = withDimensionGuard(new ClipEmbeddingProvider());
  return imageEmbeddingProvider;
}

export function createEmbeddingProvider(): EmbeddingProvider | null {
  const detected = detectEmbeddingProvider();
  if (!detected) return null;

  switch (detected) {
    case "gemini":
      return withDimensionGuard(new GeminiEmbeddingProvider(getEnvVar("GEMINI_API_KEY")!));
    case "openai":
      return withDimensionGuard(new OpenAIEmbeddingProvider(getEnvVar("OPENAI_API_KEY")!));
    case "voyage":
      return withDimensionGuard(new VoyageEmbeddingProvider(getEnvVar("VOYAGE_API_KEY")!));
    case "cohere":
      return withDimensionGuard(new CohereEmbeddingProvider(getEnvVar("COHERE_API_KEY")!));
    case "openrouter":
      return withDimensionGuard(new OpenRouterEmbeddingProvider(getEnvVar("OPENROUTER_API_KEY")!));
    case "local":
      return withDimensionGuard(new LocalEmbeddingProvider());
    default:
      return null;
  }
}

// #931-class fix: the worker's boot-time embedding probe (src/index.ts)
// used to report its result only through `bootLog`, which reaches
// stderr solely under --verbose (see the comment above `bootLog` in
// src/logger.ts). A daemon (launchd/systemd) start never sets that, so
// on a live deployment a broken embedding runtime never printed a
// single line - the corpus reached 201,102 observations at 1.1% vector
// coverage before anyone noticed. `logger.info`/`logger.warn` reach the
// daemon log unconditionally, so this reports through `logger` first,
// with `bootLog` kept alongside so --verbose still shows it in the
// compact boot summary the CLI builds from the buffer.
//
// Exported (rather than left as an inline .then()/.catch() at the call
// site in src/index.ts) so tests can call and await it directly - the
// caller dispatches this fire-and-forget, so there is no other way to
// observe its settlement from outside.
export async function reportEmbeddingProbeResult(
  embeddingProvider: EmbeddingProvider,
): Promise<void> {
  try {
    // Probe embedBatch, not embed: the indexing path
    // (vectorIndexAddBatchGuarded in search.ts) calls embedBatch, and a
    // provider can implement the two differently. Verify the shape too -
    // the guard drops any vector whose length differs from `dimensions`,
    // so a wrong-shape provider would pass a bare probe call yet index
    // nothing.
    const vectors = await embeddingProvider.embedBatch([
      "agentmemory boot probe",
    ]);
    if (
      vectors.length !== 1 ||
      vectors[0].length !== embeddingProvider.dimensions
    ) {
      throw new Error(
        `embedBatch returned ${vectors.length} vector(s) of length ` +
          `${vectors[0]?.length ?? 0}, expected 1 of length ${embeddingProvider.dimensions}`,
      );
    }
    logger.info("Embedding provider verified", {
      provider: embeddingProvider.name,
      dimensions: embeddingProvider.dimensions,
    });
    bootLog(`Embeddings: ${embeddingProvider.name} (${embeddingProvider.dimensions}d)`);
  } catch (err) {
    logger.warn(
      "Embedding provider failed boot probe - semantic search degrades to BM25-only",
      {
        provider: embeddingProvider.name,
        error: err instanceof Error ? err.message : String(err),
      },
    );
    bootLog(
      `Embeddings: ${embeddingProvider.name} FAILED - semantic search will be BM25-only. ` +
        (err instanceof Error ? err.message : String(err)),
    );
  }
}

// Wrong-dimension vectors corrupt the index silently: vector-index.ts
// returns 0 from cosineSimilarity on length mismatch instead of throwing,
// so a bad vector is stored, never matches anything, and the memory
// becomes invisible without an error. Catch it at the boundary.
export function withDimensionGuard(provider: EmbeddingProvider): EmbeddingProvider {
  const expected = provider.dimensions;
  const check = (v: Float32Array, where: string): Float32Array => {
    if (v.length !== expected) {
      throw new Error(
        `Embedding dimension mismatch in ${provider.name}.${where}: expected ${expected}, got ${v.length}`,
      );
    }
    return v;
  };
  // Preserve the provider's prototype chain so `instanceof` checks
  // against concrete classes (e.g. GeminiEmbeddingProvider) keep working.
  const wrapped = Object.create(provider) as EmbeddingProvider;
  wrapped.embed = async (t) => check(await provider.embed(t), "embed");
  wrapped.embedBatch = async (ts) => {
    const out = await provider.embedBatch(ts);
    out.forEach((v, i) => check(v, `embedBatch[${i}]`));
    return out;
  };
  if (provider.embedImage) {
    wrapped.embedImage = async (s: string) =>
      check(await provider.embedImage!(s), "embedImage");
  }
  return wrapped;
}
