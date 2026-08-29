import type { EmbeddingProvider } from "../../types.js";

type FeatureExtractor = (
  texts: string[],
  options: { pooling: string; normalize: boolean },
) => Promise<{ tolist: () => number[][] }>;

export class LocalEmbeddingProvider implements EmbeddingProvider {
  readonly name = "local";
  readonly dimensions = 384;
  private extractorPromise: Promise<FeatureExtractor> | null = null;

  async embed(text: string): Promise<Float32Array> {
    const [result] = await this.embedBatch([text]);
    return result;
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    const extractor = await this.getExtractor();
    const output = await extractor(texts, {
      pooling: "mean",
      normalize: true,
    });
    return output.tolist().map((v) => new Float32Array(v));
  }

  // Caches the in-flight load, not just the loaded extractor: the boot
  // probe (reportEmbeddingProbeResult) and a BM25 rebuild's embedding
  // queue can both hit a cold provider at once, and with only a
  // post-await cache each concurrent caller kicks off its own
  // pipeline() initialization - duplicate model download, memory, and
  // startup CPU. The promise is evicted on rejection so a transient
  // failure (an interrupted model download) is retried on the next
  // call instead of being cached until restart.
  private getExtractor(): Promise<FeatureExtractor> {
    if (!this.extractorPromise) {
      const loading = this.loadExtractor();
      this.extractorPromise = loading;
      loading.catch(() => {
        if (this.extractorPromise === loading) this.extractorPromise = null;
      });
    }
    return this.extractorPromise;
  }

  private async loadExtractor(): Promise<FeatureExtractor> {
    let transformers: typeof import("@huggingface/transformers");
    try {
      transformers = await import("@huggingface/transformers");
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ERR_MODULE_NOT_FOUND") {
        // #395: installs predating the package rename carry
        // @xenova/transformers 2.x, whose pipeline options differ. Say so
        // explicitly - the symptom is otherwise a silent 0% embed rate.
        throw new Error(
          "Local embeddings need @huggingface/transformers (>=4). " +
            "Install it with: npm install @huggingface/transformers. " +
            "The legacy @xenova/transformers 2.x package is NOT compatible " +
            "with this provider and will not be used even if present.",
        );
      }
      throw err;
    }
    return (await transformers.pipeline(
      "feature-extraction",
      "Xenova/all-MiniLM-L6-v2",
      { dtype: "q8" },
    )) as FeatureExtractor;
  }
}
