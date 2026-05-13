/**
 * Kilo Provider Extension
 *
 * Provides access to 300+ AI models via the Kilo Gateway (OpenRouter-compatible).
 * Uses device code flow for browser-based authentication.
 *
 * Usage:
 *   pi install git:github.com/mrexodia/kilo-pi-provider
 *   # Then /login
 */

import type {
  ExtensionAPI,
  OAuthCredential,
  ProviderConfig,
  ProviderModelConfig,
} from "@earendil-works/pi-coding-agent";
import { mkdirSync, readFileSync, existsSync, access } from "fs";
import { homedir } from "os";
import { join } from "path";

// =============================================================================
// Constants
// =============================================================================

const KILO_API_BASE = process.env.KILO_API_URL || "https://api.kilo.ai";
const KILO_GATEWAY_BASE = `${KILO_API_BASE}/api/gateway`;
const KILO_DEVICE_AUTH_ENDPOINT = `${KILO_API_BASE}/api/device-auth/codes`;
const POLL_INTERVAL_MS = 3000;
const KILO_TOKEN_EXPIRY_MS = 5 * 365 * 24 * 60 * 60 * 1000;
const MODELS_FETCH_TIMEOUT_MS = 10_000;
const KILO_TOS_URL = "https://kilo.ai/terms";
const KILO_PROFILE_ENDPOINT = `${KILO_API_BASE}/api/profile`;

// =============================================================================
// Balance Fetching
// =============================================================================

interface KiloBalance {
  balance?: number;
}

async function fetchKiloBalance(token: string): Promise<number | null> {
  try {
    const response = await fetch(`${KILO_PROFILE_ENDPOINT}/balance`, {
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      return null;
    }

    const data = (await response.json()) as KiloBalance;
    return data.balance ?? null;
  } catch {
    return null;
  }
}

function formatCredits(balance: number): string {
  if (balance >= 1000) {
    return `$${(balance / 1000).toFixed(1)}k`;
  } else {
    return `$${balance.toFixed(2)}`;
  }
}

/** Fire-and-forget: fetches balance in the background and updates the status bar.
 *  Uses a sequential queue so footer updates never race or show stale results.
 *  Caps at 2 pending fetches; if the queue is full the call is silently dropped. */
let creditsQueue: Promise<void> = Promise.resolve();
let creditsGeneration = 0;
let creditsPending = 0;
let creditsVisible = false;
const MAX_CREDITS_PENDING = 2;

function updateCreditsDisplay(
  ctx: {
    ui: { setStatus: (key: string, value: string | undefined) => void; theme: { fg: (kind: string, text: string) => string } };
  },
  access: string,
): void {
  if (creditsPending >= MAX_CREDITS_PENDING) return;

  const gen = ++creditsGeneration;
  creditsPending++;

  // Show placeholder immediately so the layout doesn't shift when balance loads.
  // Skip if the footer already has content (e.g. a previous balance or placeholder).
  if (!creditsVisible) {
    creditsVisible = true;
    ctx.ui.setStatus("kilo-credits", ctx.ui.theme.fg("accent", "💰 ..."));
  }

  creditsQueue = creditsQueue.then(async () => {
    creditsPending--;
    // Abort if a newer generation started or clearCreditsDisplay was called.
    if (gen !== creditsGeneration) return;
    try {
      const balance = await fetchKiloBalance(access);
      if (gen !== creditsGeneration) return;
      if (balance !== null) {
        creditsVisible = true;
        ctx.ui.setStatus(
          "kilo-credits",
          ctx.ui.theme.fg("accent", `💰 ${formatCredits(balance)}`),
        );
      }
    } catch (error) {
      if (gen !== creditsGeneration) return;
      console.warn(
        "[kilo] Failed to fetch balance:",
        error instanceof Error ? error.message : error,
      );
    }
  });
}

/** Cancel all pending balance updates and clear the footer. Call when switching away from Kilo. */
function clearCreditsDisplay(
  ctx: {
    ui: { setStatus: (key: string, value: string | undefined) => void };
  },
): void {
  creditsGeneration++;
  creditsPending = 0;
  creditsVisible = false;
  ctx.ui.setStatus("kilo-credits", undefined);
}

// =============================================================================
// Device Authorization Flow
// =============================================================================

interface DeviceAuthResponse {
  code: string;
  verificationUrl: string;
  expiresIn: number;
}

interface DeviceAuthPollResponse {
  status: "pending" | "approved" | "denied" | "expired";
  token?: string;
  userEmail?: string;
}

function abortableSleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new Error("Login cancelled"));
      return;
    }
    const timeout = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timeout);
        reject(new Error("Login cancelled"));
      },
      { once: true },
    );
  });
}

function isTokenFresh(tokenExpiry: number | null, now: number): boolean {
  return tokenExpiry != null && tokenExpiry > now;
}

const refreshPromises = new Map<string, Promise<OAuthCredential>>();

// Kilo uses long-lived JWT tokens from device auth flow - no separate refresh needed.
// The token is used directly as Bearer token for API calls.
async function refreshAccessToken(
  currentCredentials: OAuthCredential,
): Promise<OAuthCredential> {
  const cacheKey = "kilo";
  const cached = refreshPromises.get(cacheKey);
  if (cached) return cached;

  const promise = (async () => {
    // Kilo tokens are JWTs from device auth that work directly as Bearer tokens.
    // Just extend the expiry to match Kilo's JWT expiry (~6 years).
    return {
      ...currentCredentials,
      expires: Date.now() + KILO_TOKEN_EXPIRY_MS,
    };
  })();

  refreshPromises.set(cacheKey, promise);
  try {
    return await promise;
  } finally {
    refreshPromises.delete(cacheKey);
  }
}

async function initiateDeviceAuth(): Promise<DeviceAuthResponse> {
  const response = await fetch(KILO_DEVICE_AUTH_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    if (response.status === 429) {
      throw new Error(
        "Too many pending authorization requests. Please try again later.",
      );
    }
    throw new Error(
      `Failed to initiate device authorization: ${response.status}`,
    );
  }

  return (await response.json()) as DeviceAuthResponse;
}

async function pollDeviceAuth(code: string): Promise<DeviceAuthPollResponse> {
  const response = await fetch(`${KILO_DEVICE_AUTH_ENDPOINT}/${code}`);

  if (response.status === 202) return { status: "pending" };
  if (response.status === 403) return { status: "denied" };
  if (response.status === 410) return { status: "expired" };

  if (!response.ok) {
    throw new Error(`Failed to poll device authorization: ${response.status}`);
  }

  return (await response.json()) as DeviceAuthPollResponse;
}

// =============================================================================
// Dynamic Model Loading
// =============================================================================

interface KiloModel {
  id: string;
  name: string;
  context_length: number;
  max_completion_tokens?: number | null;
  pricing?: {
    prompt?: string | null;
    completion?: string | null;
    input_cache_write?: string | null;
    input_cache_read?: string | null;
  };
  architecture?: {
    input_modalities?: string[] | null;
    output_modalities?: string[] | null;
  };
  top_provider?: { max_completion_tokens?: number | null };
  supported_parameters?: string[];
  preferredIndex?: number;
  isFree?: boolean;
  openclaw?: {
    api_adapter?: string;
  };
  opencode?: {
    ai_sdk_provider?: string;
    family?: string;
    prompt?: string;
    variants?: {
      [level: string]: {
        reasoning?: { enabled: boolean; effort: string };
        verbosity?: string;
      };
    };
  };
}

function formatPrice(price: string | null | undefined): string {
  if (!price) return "";
  const num = parseFloat(price) * 1_000_000;
  if (num <= 0 || isNaN(num)) return "";
  if (num >= 1_000) return `$${(num / 1000).toFixed(1)}k`;
  // Always format in dollars. toFixed(10) gives enough precision,
  // then parseFloat+toString strips meaningless trailing zeros.
  const formatted = parseFloat(num.toFixed(10)).toString();
  return `$${formatted}`;
}

function formatModelPriceLabel(m: KiloModel): string {
  if (m.isFree) return "\n";
  const items: { label: string; value: string }[] = [];
  const input = formatPrice(m.pricing?.prompt);
  if (input) items.push({ label: "I", value: input });
  const output = formatPrice(m.pricing?.completion);
  if (output) items.push({ label: "O", value: output });
  const cacheRead = formatPrice(m.pricing?.input_cache_read);
  if (cacheRead) items.push({ label: "CR", value: cacheRead });
  const cacheWrite = formatPrice(m.pricing?.input_cache_write);
  if (cacheWrite) items.push({ label: "CW", value: cacheWrite });
  if (items.length === 0) return "\n";
  const parts = items.map(i => `${i.label}:${i.value.slice(0, 10).padEnd(10)}`);
  return `\n  💵 ${parts.join("  ")}`;
}

function parsePrice(price: string | null | undefined): number {
  if (!price) return 0;
  const parsed = parseFloat(price);
  if (isNaN(parsed)) return 0;
  // OpenRouter prices are per-token; Pi expects per-million-token
  return parsed * 1_000_000;
}

/**
 * Build compat settings for Kilo models.
 *
 * Kilo uses OpenRouter-format model IDs (e.g. openai/gpt-4o, anthropic/claude-4-sonnet),
 * but the auto-detection in detectCompat() doesn't recognize the Kilo baseUrl.
 * We override compat to match what OpenRouter models expect, mimicking the
 * official detectCompat() for provider === "openrouter".
 */
function getKiloCompat(m: KiloModel): ProviderModelConfig["compat"] {
  const prefix = m.id.split("/")[0] ?? "";
  const sdkProvider = m.opencode?.ai_sdk_provider;

  // Derive compat settings from both the SDK provider and the ID prefix
  // independently, then merge them together.
  const fromProvider = getCompatFromSdkProvider(sdkProvider);
  const fromPrefix = getCompatFromPrefix(prefix);

  return {
    ...fromProvider,
    ...fromPrefix,
  };
}

function getCompatFromSdkProvider(
  sdkProvider: string | undefined,
): Partial<ProviderModelConfig["compat"]> {
  switch (sdkProvider) {
    case "anthropic":
      return { cacheControlFormat: "anthropic" };
    case "alibaba":
      return { supportsDeveloperRole: false };
    default:
      return {};
  }
}

function getCompatFromPrefix(
  prefix: string,
): Partial<ProviderModelConfig["compat"]> {
  switch (prefix) {
    case "deepseek":
      return {
        thinkingFormat: "deepseek",
        requiresReasoningContentOnAssistantMessages: true,
      };
    case "z-ai":
      return {
        thinkingFormat: "zai",
        supportsReasoningEffort: false,
      };
    case "moonshotai":
      return {
        thinkingFormat: "openai",
        supportsReasoningEffort: false,
        supportsStore: false,
        supportsDeveloperRole: false,
        maxTokensField: "max_tokens",
        supportsStrictMode: false,
      };
    case "qwen":
      return { supportsDeveloperRole: false };
    case "anthropic":
      return { cacheControlFormat: "anthropic" };
    default:
      return {};
  }
}

// Map from Kilo variant keys to pi thinking level keys.
const VARIANT_TO_PI_LEVEL: Record<string, string> = {
  none: "off",
  instant: "minimal",
  low: "low",
  medium: "medium",
  high: "high",
  max: "xhigh",
  xhigh: "xhigh",
  thinking: "medium",
};

/**
 * Build per-model overrides (thinking levels, etc.) from Kilo model metadata.
 */
function getKiloModelOverrides(m: KiloModel): Partial<ProviderModelConfig> {
  // Use opencode.variants if available for reasoning level mapping
  const variants = m.opencode?.variants;
  if (variants) {
    const thinkingLevelMap: Record<string, string | null> = {};
    let hasNonNone = false;
    for (const [level, config] of Object.entries(variants)) {
      const piLevel = VARIANT_TO_PI_LEVEL[level];
      if (!piLevel) continue;
      const effort = config.reasoning?.effort;
      if (piLevel === "off") {
        // The "off" level means no reasoning; mark as unsupported.
        thinkingLevelMap[piLevel] = null;
      } else {
        // Use the variant's effort value directly (even "none").
        thinkingLevelMap[piLevel] = effort ?? null;
        if (effort) hasNonNone = true;
      }
    }
    const result: Partial<ProviderModelConfig> = {
      thinkingLevelMap,
    };
    if (hasNonNone) {
      return result;
    }
  }

  // Fallback to hardcoded mappings for known models
  switch (m.id) {
    case "deepseek/deepseek-v4-pro":
    case "deepseek/deepseek-v4-flash":
      return {
        thinkingLevelMap: {
          off: null,
          minimal: null,
          low: null,
          medium: null,
          xhigh: "xhigh",
        },
      };
    default:
      return {};
  }
}

function mapOpenRouterModel(m: KiloModel): ProviderModelConfig {
  const inputModalities = m.architecture?.input_modalities ?? ["text"];
  const supportsImages = inputModalities.includes("image");
  // Reasoning support: check both the API's supported_parameters and
  // the presence of opencode.variants with reasoning effort levels.
  const hasReasoningParam =
    m.supported_parameters?.includes("reasoning") ?? false;
  const hasReasoningVariants = Object.entries(
    m.opencode?.variants ?? {},
  ).some(([level, v]) => {
    const piLevel = VARIANT_TO_PI_LEVEL[level];
    return (
      piLevel &&
      piLevel !== "off" &&
      v.reasoning?.effort &&
      v.reasoning.effort !== "none"
    );
  });
  const supportsReasoning = hasReasoningParam || hasReasoningVariants;
  const maxTokens =
    m.top_provider?.max_completion_tokens ??
    m.max_completion_tokens ??
    Math.ceil(m.context_length * 0.2);

  const overrides = getKiloModelOverrides(m);

  // Map Kilo/OpenRouter api_adapter to pi Api type.
  // Default is "openai-completions" for most models.
  let modelApi: ProviderModelConfig["api"]; // Api type
  const adapter = m.openclaw?.api_adapter;
  if (adapter === "openai-responses") {
    modelApi = adapter;
  }

  // Also check opencode metadata for Anthropic models.
  // OpenRouter/Kilo marks Anthropic models with ai_sdk_provider or prompt "anthropic".
  if (!modelApi) {
    const oc = m.opencode;
    if (
      oc?.ai_sdk_provider === "anthropic" ||
      oc?.prompt === "anthropic" ||
      oc?.family === "claude"
    ) {
      modelApi = "anthropic-messages";
    }
  }

  // ai_sdk_provider can also directly specify the API type (openai, anthropic, alibaba)
  if (!modelApi && m.opencode?.ai_sdk_provider) {
    const provider = m.opencode.ai_sdk_provider;
    if (provider === "anthropic") {
      modelApi = "anthropic-messages";
    } else if (provider === "openai") {
      modelApi = "openai-completions";
    }
    // alibaba (Qwen) uses default openai-completions format
  }

  return {
    id: m.id,
    name: `${m.name}${formatModelPriceLabel(m)}`,
    api: modelApi,
    reasoning: supportsReasoning,
    input: supportsImages ? ["text", "image"] : ["text"],
    cost: {
      input: parsePrice(m.pricing?.prompt),
      output: parsePrice(m.pricing?.completion),
      cacheRead: parsePrice(m.pricing?.input_cache_read),
      cacheWrite: parsePrice(m.pricing?.input_cache_write),
    },
    contextWindow: m.context_length,
    maxTokens: maxTokens,
    compat: getKiloCompat(m),
    ...overrides,
  };
}

// States
let modelList: ProviderModelConfig[] = [];
let modelUpdated = false;

const MODELS_CACHE_FILE = `pi-kilo-models-cache.json`;
const TOS_CACHE_FILE = `pi-kilo-tos-shown.json`;

async function saveModelsToCache(models: KiloModel[], free: boolean) {
  try {
    const { writeFile } = await import("fs/promises");
    const cacheDir = join(homedir(), ".cache", "pi");
    mkdirSync(cacheDir, { recursive: true });
    const cachePath = join(cacheDir, MODELS_CACHE_FILE);
    await writeFile(cachePath, JSON.stringify({ models, free }), "utf-8");
  } catch (error) {
    console.warn(
      "[kilo] Failed to save models cache:",
      error instanceof Error ? error.message : error,
    );
  }
}

/** Synchronous: reads from local disk cache. Fast, no I/O wait. */
function loadModelsFromCache(): {
  models: KiloModel[];
  free: boolean;
} | null {
  try {
    const cacheDir = join(homedir(), ".cache", "pi");
    mkdirSync(cacheDir, { recursive: true });
    const cachePath = join(cacheDir, MODELS_CACHE_FILE);
    if (!existsSync(cachePath)) return null;
    const data = readFileSync(cachePath, "utf-8");
    const parsed = JSON.parse(data);
    if (!parsed || !Array.isArray(parsed.models)) return null;
    return parsed;
  } catch {
    return null;
  }
}

/** Synchronous: reads tosShown from cache. */
function loadTosShownFromCache(): boolean {
  try {
    const cacheDir = join(homedir(), ".cache", "pi");
    mkdirSync(cacheDir, { recursive: true });
    const cachePath = join(cacheDir, TOS_CACHE_FILE);
    if (!existsSync(cachePath)) return false;
    const data = readFileSync(cachePath, "utf-8");
    return data === "true";
  } catch {
    return false;
  }
}

/** Persist tosShown to cache file. */
function saveTosShownToCache(value: boolean): void {
  try {
    const cacheDir = join(homedir(), ".cache", "pi");
    mkdirSync(cacheDir, { recursive: true });
    const cachePath = join(cacheDir, TOS_CACHE_FILE);
    import("fs/promises").then(({ writeFile }) =>
      writeFile(cachePath, String(value), "utf-8"),
    ).catch(() => {
      // Silently fail
    });
  } catch {
    // Silently fail
  }
}

function updateModels(models: KiloModel[], free: boolean) {
  modelList.length = 0;
  modelList.push(
    ...models
      .filter((m) => {
        const outputMods = m.architecture?.output_modalities ?? [];
        if (outputMods.includes("image")) return false;
        if (free && !m.isFree) return false;
        return true;
      })
      .sort(
        (a, b) =>
          (a.preferredIndex || Infinity) - (b.preferredIndex || Infinity),
      )
      .map(mapOpenRouterModel),
  );
}

async function updateKiloModels(free: boolean) {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "User-Agent": "pi-kilo-provider",
  };

  const response = await fetch(`${KILO_GATEWAY_BASE}/models`, {
    headers,
    signal: AbortSignal.timeout(MODELS_FETCH_TIMEOUT_MS),
  });

  if (!response.ok) {
    throw new Error(
      `Failed to fetch models: ${response.status} ${response.statusText}`,
    );
  }

  const json = (await response.json()) as { data?: KiloModel[] };
  if (!json.data || !Array.isArray(json.data)) {
    throw new Error("Invalid models response: missing data array");
  }

  await saveModelsToCache(json.data, free);
  updateModels(json.data, free);
  modelUpdated = true;
}

// =============================================================================
// Provider Config
// =============================================================================

const KILO_PROVIDER_CONFIG: Partial<ProviderConfig> = {
  baseUrl: KILO_GATEWAY_BASE,
  api: "openai-completions",
  authHeader: true,
  apiKey: "<none>", // no api key required to use free models 
  headers: {
    "X-KILOCODE-EDITORNAME": "Pi",
    "User-Agent": "pi-kilo-provider",
  },
};

// =============================================================================
// Extension Entry Point
// =============================================================================

export default async function (pi: ExtensionAPI) {
  // Load cached models synchronously so models are available at registration time.
  const loaded = loadModelsFromCache();
  if (loaded) {
    updateModels(loaded.models, loaded.free);
  } else {
    // First run: no cache. We must fetch models so pi has something to register.
    // This only blocks on the very first launch.
    try {
      await updateKiloModels(true);
    } catch (error) {
      console.warn(
        "[kilo] Failed to fetch initial models:",
        error instanceof Error ? error.message : error,
      );
    }
  }

  // Always refresh models from the network in the background.
  // Preserve the cached free/all mode to avoid downgrading from all→free on restart.
  updateKiloModels(loaded?.free ?? true).catch((error) => {
    console.warn(
      "[kilo] Failed to fetch initial models:",
      error instanceof Error ? error.message : error,
    );
  });

  pi.registerProvider("kilo", {
    ...KILO_PROVIDER_CONFIG,
    models: modelList,
    oauth: {
      name: "Kilo",
      login: async (callbacks) => {
        async function loginKilo(
          cb: typeof callbacks,
        ): Promise<OAuthCredential> {
          cb.onProgress?.("Initiating device authorization...");
          const authData = await initiateDeviceAuth();
          const { code, verificationUrl, expiresIn } = authData;

          cb.onAuth({
            url: verificationUrl,
            instructions: `Enter code: ${code}`,
          });

          cb.onProgress?.("Waiting for browser authorization...");

          const deadline = Date.now() + expiresIn * 1000;
          while (Date.now() < deadline) {
            if (cb.signal?.aborted) {
              throw new Error("Login cancelled");
            }

            await abortableSleep(POLL_INTERVAL_MS, cb.signal);

            const result = await pollDeviceAuth(code);

            if (result.status === "approved") {
              if (!result.token) {
                throw new Error("Authorization approved but no token received");
              }
              cb.onProgress?.("Login successful!");
              // Kilo tokens are long-lived JWTs - set expiry to ~6 years
              return {
                type: "oauth",
                refresh: result.token,
                access: result.token,
                expires: Date.now() + KILO_TOKEN_EXPIRY_MS,
              };
            }

            if (result.status === "denied") {
              throw new Error("Authorization denied by user.");
            }

            if (result.status === "expired") {
              throw new Error("Authorization code expired. Please try again.");
            }
          }

          throw new Error("Authentication timed out. Please try again.");
        }

        const cred = await loginKilo(callbacks);
        try {
          await updateKiloModels(!cred?.access);
        } catch (error) {
          console.warn(
            "[kilo] Failed to fetch models on session start:",
            error instanceof Error ? error.message : error,
          );
        }
        return cred;
      },
      refreshToken: async (credentials: OAuthCredential) => {
        if (isTokenFresh(credentials.expires, Date.now())) {
          return credentials;
        }
        try {
          const newCreds = await refreshAccessToken(credentials);
          return newCreds;
        } catch (error) {
          console.warn("[kilo] Token refresh failed, keeping existing credentials:", error instanceof Error ? error.message : error);
          // Return existing credentials with extended expiry to avoid repeated refresh attempts
          return {
            ...credentials,
            expires: Date.now() + KILO_TOKEN_EXPIRY_MS,
          };
        }
      },
      getApiKey: (cred) => cred.access,
    },
  });

  // After session starts, pre-fetch all models if already logged in so
  // modifyModels has data to work with. Also fetch and display credits.
  pi.on("session_start", async (_event, ctx) => {
    const cred = ctx.modelRegistry.authStorage.get("kilo");

    let updating = false;
    if (!cred !== loaded?.free || !modelUpdated) {
      updateKiloModels(!cred).catch((error) => {
        console.warn(
          "[kilo] Failed to fetch models on session start:",
          error instanceof Error ? error.message : error,
        );
      }).then(() => {
        ctx.modelRegistry.refresh();
        // Update current model in case any capabilities are changed
        // this is most likely happen to router models (e.g. kilo-auto)
        const currentModel = ctx.model;
        if (currentModel && currentModel.provider === "kilo") {
          const model = ctx.modelRegistry.find("kilo", currentModel.id);
          if (model) {
            Object.assign(currentModel, model);
          }
        }
      });
      updating = true;
    }

    // Clear credits if not logged in or not using Kilo models
    if (cred?.type !== "oauth" || ctx.model?.provider !== "kilo") {
      clearCreditsDisplay(ctx as any);
      return;
    }

    // Fetch and display credits balance (fire-and-forget)
    updateCreditsDisplay(ctx as any, cred.access);
  });

  // Update credits display when model changes to a Kilo model
  pi.on("model_select", async (event, ctx) => {
    const cred = ctx.modelRegistry.authStorage.get("kilo");

    // Clear credits if not logged in or not using Kilo models
    if (cred?.type !== "oauth" || ctx.model?.provider !== "kilo") {
      clearCreditsDisplay(ctx as any);
      return;
    }

    // Fetch and display credits balance (fire-and-forget)
    updateCreditsDisplay(ctx as any, cred.access);
  });

  // Refresh credits after each turn
  pi.on("turn_end", async (_event, ctx) => {
    const cred = ctx.modelRegistry.authStorage.get("kilo");

    // Clear credits if not logged in or not using Kilo models
    if (cred?.type !== "oauth" || ctx.model?.provider !== "kilo") {
      clearCreditsDisplay(ctx as any);
      return;
    }

    // Fetch and display credits balance (fire-and-forget)
    updateCreditsDisplay(ctx as any, cred.access);
  });

  // On first use of a Kilo model without login, print ToS notice.
  // Persisted to cache so it only shows once per user.
  let tosShown = loadTosShownFromCache();

  pi.on("before_agent_start", async (_event, ctx) => {
    if (tosShown) return;
    if (ctx.model?.provider !== "kilo") return;

    const cred = ctx.modelRegistry.authStorage.get("kilo");
    if (cred?.type === "oauth") {
      tosShown = true;
      saveTosShownToCache(true);
      return;
    }

    tosShown = true;
    saveTosShownToCache(true);

    return {
      message: {
        customType: "kilo",
        content: `By using Kilo, you agree to the Terms of Service: ${KILO_TOS_URL}`,
        display: true,
      },
    };
  });
}
