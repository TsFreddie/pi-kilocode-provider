# Kilo Provider for Pi

A Kilo provider extension for [Pi](https://pi.dev) — the coding agent CLI.
Access 300+ AI models through the [Kilo Gateway](https://kilo.ai) (OpenRouter-compatible).

## Features

- **300+ models** — dynamically fetched from the Kilo Gateway
- **`/login` support** — authenticates via browser; shows your Kilo credit balance in the status bar
- **Pricing in model picker** — per-model pricing shown inline:
  ```
    Model Name: GPT-4o
    💵 I:$2.50      O:$10        CR:$1.25      CW:$2.50
  ```
  Input (I), output (O), cache read (CR), and cache write (CW) prices per million tokens.
- **Free models** available without login
- **Model caching** — model list is cached locally for fast startup
- **Smart compat** — handles provider-specific quirks (DeepSeek thinking, Anthropic cache, etc.)

## Prerequisites

Install [Pi](https://pi.dev):

```bash
npm install -g @earendil-works/pi-coding-agent
```

## Installation

```bash
pi install git:github.com/TsFreddie/pi-kilocode-provider
```

## Usage

Start Pi as usual:

```bash
pi
```

Free models are available immediately. To access all models, log in with your Kilo account:

```
/login
```

This opens your browser for authorization. Once approved, your credit balance appears in the status bar and all models become available in the model selector (`Ctrl+L`).
