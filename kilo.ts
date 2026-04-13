/**
 * Kilo Provider Extension
 *
 * Provides access to 300+ AI models via the Kilo Gateway (OpenRouter-compatible).
 * Uses device code flow for browser-based authentication.
 *
 * Usage:
 *   pi install git:github.com/mrexodia/kilo-pi-provider
 *   # Then /login kilo, or set KILO_API_KEY=...
 */

import type {
  ExtensionAPI,
  OAuthCredential,
  ProviderModelConfig,
} from "@mariozechner/pi-coding-agent";
import { visibleWidth } from "@mariozechner/pi-tui";
import { mkdirSync } from "fs";

// =============================================================================
// Constants
// =============================================================================

const KILO_API_BASE = process.env.KILO_API_URL || "https://api.kilo.ai";
const KILO_GATEWAY_BASE = `${KILO_API_BASE}/api/gateway`;
const KILO_DEVICE_AUTH_ENDPOINT = `${KILO_API_BASE}/api/device-auth/codes`;
const POLL_INTERVAL_MS = 3000;
// Kilo JWT tokens expire in 5 years, use same expiry
const KILO_TOKEN_EXPIRY_MS = 5 * 365.25 * 24 * 60 * 60 * 1000;
const MODELS_FETCH_TIMEOUT_MS = 10_000;
const KILO_TOS_URL = "https://kilo.ai/terms";
const KILO_PROFILE_ENDPOINT = `${KILO_API_BASE}/api/profile`;

const EAGER_REFRESH_THRESHOLD_MS = 5 * 60 * 1000;

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
  return tokenExpiry != null && tokenExpiry > now + EAGER_REFRESH_THRESHOLD_MS;
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
    console.log("[kilo] Token refresh requested - Kilo uses long-lived JWTs, no refresh needed");
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
}

function parsePrice(price: string | null | undefined): number {
  if (!price) return 0;
  const parsed = parseFloat(price);
  if (isNaN(parsed)) return 0;
  // OpenRouter prices are per-token; Pi expects per-million-token
  return parsed * 1_000_000;
}

function mapOpenRouterModel(m: KiloModel): ProviderModelConfig {
  const inputModalities = m.architecture?.input_modalities ?? ["text"];
  const supportsImages = inputModalities.includes("image");
  const supportsReasoning =
    m.supported_parameters?.includes("reasoning") ?? false;
  const maxTokens =
    m.top_provider?.max_completion_tokens ??
    m.max_completion_tokens ??
    Math.ceil(m.context_length * 0.2);

  return {
    id: m.id,
    name: m.name,
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
    compat: {
      supportsDeveloperRole: false,
    },
  };
}

// States
let modelList: ProviderModelConfig[] = [];
let modelUpdated = false;

const MODELS_CACHE_FILE = `pi-kilo-models-cache.json`;

async function saveModelsToCache(models: KiloModel[], free: boolean) {
  try {
    const { writeFileSync } = await import("fs");
    const { homedir } = await import("os");
    const { join } = await import("path");
    const cacheDir = join(homedir(), ".cache", "pi");
    mkdirSync(cacheDir, { recursive: true });
    const cachePath = join(cacheDir, MODELS_CACHE_FILE);
    writeFileSync(cachePath, JSON.stringify({ models, free }), "utf-8");
  } catch (error) {
    console.warn(
      "[kilo] Failed to save models cache:",
      error instanceof Error ? error.message : error,
    );
  }
}

async function loadModelsFromCache(): Promise<{
  models: KiloModel[];
  free: boolean;
} | null> {
  try {
    const { readFileSync, existsSync } = await import("fs");
    const { homedir } = await import("os");
    const { join } = await import("path");
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

const KILO_PROVIDER_CONFIG = {
  baseUrl: KILO_GATEWAY_BASE,
  apiKey: "KILO_API_KEY",
  api: "openai-completions" as const,
  headers: {
    "X-KILOCODE-EDITORNAME": "Pi",
    "User-Agent": "pi-kilo-provider",
  },
};

// =============================================================================
// Extension Entry Point
// =============================================================================

export default async function (pi: ExtensionAPI) {
  const loaded = await loadModelsFromCache();
  if (!loaded) {
    try {
      await updateKiloModels(true);
    } catch (error) {
      console.warn(
        "[kilo] Failed to fetch initial models:",
        error instanceof Error ? error.message : error,
      );
    }
  } else {
    updateModels(loaded.models, loaded.free);
  }

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
              console.log(result);

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

            const remaining = Math.ceil((deadline - Date.now()) / 1000);
            cb.onProgress?.(
              `Waiting for browser authorization... (${remaining}s remaining)`,
            );
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

    // Clear credits if not logged in or not using Kilo models
    if (cred?.type !== "oauth" || ctx.model?.provider !== "kilo") {
      ctx.ui.setStatus("kilo-credits", undefined);
      return;
    }

    // Only fetch models if we haven't already done so
    if (!modelUpdated) {
      try {
        await updateKiloModels(!cred);
      } catch (error) {
        console.warn(
          "[kilo] Failed to fetch models on session start:",
          error instanceof Error ? error.message : error,
        );
      }
      ctx.modelRegistry.refresh();
      // Update current model in case any capabilities are changed
      // this is most likely happen to router models (e.g. kilo-auto)
      const currentModel = ctx.model;
      if (currentModel) {
        if (currentModel.provider !== "kilo") return;
        const model = ctx.modelRegistry.find("kilo", currentModel.id);
        if (model) {
          Object.assign(currentModel, model);
        }
      }
    }

    // Fetch and display credits balance
    try {
      const balance = await fetchKiloBalance(cred.access);
      if (balance !== null) {
        ctx.ui.setStatus(
          "kilo-credits",
          ctx.ui.theme.fg("accent", `💰 ${formatCredits(balance)}`),
        );
      }
    } catch (error) {
      console.warn(
        "[kilo] Failed to fetch balance:",
        error instanceof Error ? error.message : error,
      );
    }
  });

  // Update credits display when model changes to a Kilo model
  pi.on("model_select", async (event, ctx) => {
    const cred = ctx.modelRegistry.authStorage.get("kilo");

    // Clear credits if not logged in or not using Kilo models
    if (cred?.type !== "oauth" || ctx.model?.provider !== "kilo") {
      ctx.ui.setStatus("kilo-credits", undefined);
      return;
    }

    try {
      const balance = await fetchKiloBalance(cred.access);
      if (balance !== null) {
        ctx.ui.setStatus(
          "kilo-credits",
          ctx.ui.theme.fg("accent", `💰 ${formatCredits(balance)}`),
        );
      }
    } catch (error) {
      console.warn(
        "[kilo] Failed to fetch balance on model select:",
        error instanceof Error ? error.message : error,
      );
    }
  });

  // Refresh credits after each turn
  pi.on("turn_end", async (_event, ctx) => {
    const cred = ctx.modelRegistry.authStorage.get("kilo");

    // Clear credits if not logged in or not using Kilo models
    if (cred?.type !== "oauth" || ctx.model?.provider !== "kilo") {
      ctx.ui.setStatus("kilo-credits", undefined);
      return;
    }

    try {
      const balance = await fetchKiloBalance(cred.access);
      if (balance !== null) {
        ctx.ui.setStatus(
          "kilo-credits",
          ctx.ui.theme.fg("accent", `💰 ${formatCredits(balance)}`),
        );
      }
    } catch (error) {
      console.warn(
        "[kilo] Failed to fetch balance on turn end:",
        error instanceof Error ? error.message : error,
      );
    }
  });

  // On first use of a Kilo model without login, print ToS notice.
  let tosShown = false;

  pi.on("before_agent_start", async (_event, ctx) => {
    if (tosShown) return;
    if (ctx.model?.provider !== "kilo") return;

    const cred = ctx.modelRegistry.authStorage.get("kilo");
    if (cred?.type === "oauth") {
      tosShown = true;
      return;
    }

    tosShown = true;

    return {
      message: {
        customType: "kilo",
        content: `By using Kilo, you agree to the Terms of Service: ${KILO_TOS_URL}`,
        display: true,
      },
    };
  });
}
