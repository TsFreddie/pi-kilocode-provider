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
const KILO_TOKEN_REFRESH_ENDPOINT = `${KILO_API_BASE}/auth/device/token`;
const POLL_INTERVAL_MS = 3000;
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

interface TokenRefreshResponse {
  access_token: string;
  refresh_token: string;
  expires_in: number;
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

async function refreshAccessToken(
  currentCredentials: OAuthCredential,
): Promise<OAuthCredential> {
  const cacheKey = "kilo";
  const cached = refreshPromises.get(cacheKey);
  if (cached) return cached;

  const promise = (async () => {
    const response = await fetch(KILO_TOKEN_REFRESH_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        grant_type: "refresh_token",
        refresh_token: currentCredentials.refresh,
        client_id: "opencode-cli",
      }),
    });

    if (!response.ok) {
      throw new Error(`Token refresh failed: ${response.status}`);
    }

    const data = (await response.json()) as TokenRefreshResponse;
    const newCreds: OAuthCredential = {
      type: "oauth",
      access: data.access_token,
      refresh: data.refresh_token,
      expires: Date.now() + data.expires_in * 1000,
    };
    return newCreds;
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
    mkdirSync(cacheDir, {recursive: true});
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
    mkdirSync(cacheDir, {recursive: true});
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
              return {
                type: "oauth",
                refresh: result.token,
                access: result.token,
                expires: Date.now() + 3600 * 1000,
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
        const newCreds = await refreshAccessToken(credentials);
        return newCreds;
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
    if (event.model?.provider !== "kilo") return;

    const cred = ctx.modelRegistry.authStorage.get("kilo");
    if (cred?.type !== "oauth") return;

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
    if (cred?.type !== "oauth") return;

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

  // Use custom footer to show credits inline with token stats
  pi.on("session_start", async (_event, ctx) => {
    ctx.ui.setFooter((tui, theme, footerData) => {
      const unsubBranch = footerData.onBranchChange(() => tui.requestRender());

      const formatTokens = (count: number): string => {
        if (count < 1000) return count.toString();
        if (count < 10000) return `${(count / 1000).toFixed(1)}k`;
        if (count < 1000000) return `${Math.round(count / 1000)}k`;
        if (count < 10000000) return `${(count / 1000000).toFixed(1)}M`;
        return `${Math.round(count / 1000000)}M`;
      };

      return {
        dispose() {
          unsubBranch();
        },
        invalidate() {},
        render(width: number): string[] {
          const model = ctx.model;

          // Match built-in footer totals: all assistant messages across all entries
          let totalInput = 0;
          let totalOutput = 0;
          let totalCacheRead = 0;
          let totalCacheWrite = 0;
          let totalCost = 0;
          for (const entry of ctx.sessionManager.getEntries()) {
            if (
              entry.type === "message" &&
              entry.message.role === "assistant"
            ) {
              totalInput += entry.message.usage.input;
              totalOutput += entry.message.usage.output;
              totalCacheRead += entry.message.usage.cacheRead;
              totalCacheWrite += entry.message.usage.cacheWrite;
              totalCost += entry.message.usage.cost.total;
            }
          }

          // Match built-in context usage behavior
          const contextUsage = ctx.getContextUsage();
          const contextWindow =
            contextUsage?.contextWindow ?? model?.contextWindow ?? 0;
          const contextPercentValue = contextUsage?.percent ?? 0;
          const contextPercent =
            contextUsage?.percent !== null
              ? contextPercentValue.toFixed(1)
              : "?";

          // Build pwd line like built-in (path + branch + session name)
          let pwd = process.cwd();
          const home = process.env.HOME || process.env.USERPROFILE;
          if (home && pwd.startsWith(home)) pwd = `~${pwd.slice(home.length)}`;
          const branch = footerData.getGitBranch();
          if (branch) pwd = `${pwd} (${branch})`;
          const sessionName = ctx.sessionManager.getSessionName();
          if (sessionName) pwd = `${pwd} • ${sessionName}`;

          if (pwd.length > width) {
            const half = Math.floor(width / 2) - 2;
            if (half > 1) {
              pwd = `${pwd.slice(0, half)}...${pwd.slice(-(half - 1))}`;
            } else {
              pwd = pwd.slice(0, Math.max(1, width));
            }
          }

          const statsParts: string[] = [];
          if (totalInput) statsParts.push(`↑${formatTokens(totalInput)}`);
          if (totalOutput) statsParts.push(`↓${formatTokens(totalOutput)}`);
          if (totalCacheRead)
            statsParts.push(`R${formatTokens(totalCacheRead)}`);
          if (totalCacheWrite)
            statsParts.push(`W${formatTokens(totalCacheWrite)}`);

          const usingSubscription = model
            ? ctx.modelRegistry.isUsingOAuth(model)
            : false;
          if (totalCost || usingSubscription) {
            statsParts.push(
              `$${totalCost.toFixed(3)}${usingSubscription ? " (sub)" : ""}`,
            );
          }

          const autoIndicator = " (auto)";
          const contextPercentDisplay =
            contextPercent === "?"
              ? `?/${formatTokens(contextWindow)}${autoIndicator}`
              : `${contextPercent}%/${formatTokens(contextWindow)}${autoIndicator}`;

          let contextPercentStr: string;
          if (contextPercentValue > 90) {
            contextPercentStr = theme.fg("error", contextPercentDisplay);
          } else if (contextPercentValue > 70) {
            contextPercentStr = theme.fg("warning", contextPercentDisplay);
          } else {
            contextPercentStr = contextPercentDisplay;
          }
          statsParts.push(contextPercentStr);

          // Inject credits inline on the main stats line
          const creditsStatus = footerData
            .getExtensionStatuses()
            .get("kilo-credits");
          if (creditsStatus) statsParts.push(creditsStatus);

          let statsLeft = statsParts.join(" ");
          let statsLeftWidth = visibleWidth(statsLeft);

          // Right side: model + thinking + provider like built-in
          const modelName = model?.id || "no-model";
          let rightSideWithoutProvider = modelName;
          if (model?.reasoning) {
            const thinkingLevel = pi.getThinkingLevel() || "off";
            rightSideWithoutProvider =
              thinkingLevel === "off"
                ? `${modelName} • thinking off`
                : `${modelName} • ${thinkingLevel}`;
          }

          let rightSide = rightSideWithoutProvider;
          if (footerData.getAvailableProviderCount() > 1 && model) {
            rightSide = `(${model.provider}) ${rightSideWithoutProvider}`;
            if (statsLeftWidth + 2 + visibleWidth(rightSide) > width) {
              rightSide = rightSideWithoutProvider;
            }
          }

          if (statsLeftWidth > width) {
            const plainStatsLeft = statsLeft.replace(/\x1b\[[0-9;]*m/g, "");
            statsLeft = `${plainStatsLeft.substring(0, width - 3)}...`;
            statsLeftWidth = visibleWidth(statsLeft);
          }

          const rightSideWidth = visibleWidth(rightSide);
          const totalNeeded = statsLeftWidth + 2 + rightSideWidth;

          let statsLine: string;
          if (totalNeeded <= width) {
            const padding = " ".repeat(width - statsLeftWidth - rightSideWidth);
            statsLine = statsLeft + padding + rightSide;
          } else {
            const availableForRight = width - statsLeftWidth - 2;
            if (availableForRight > 3) {
              const plainRight = rightSide.replace(/\x1b\[[0-9;]*m/g, "");
              const truncatedRight = plainRight.substring(0, availableForRight);
              const padding = " ".repeat(
                width - statsLeftWidth - truncatedRight.length,
              );
              statsLine = statsLeft + padding + truncatedRight;
            } else {
              statsLine = statsLeft;
            }
          }

          const dimStatsLeft = theme.fg("dim", statsLeft);
          const remainder = statsLine.slice(statsLeft.length);
          const dimRemainder = theme.fg("dim", remainder);

          return [theme.fg("dim", pwd), dimStatsLeft + dimRemainder];
        },
      };
    });
  });
}
