use anyhow::{anyhow, Context, Result};
use reqwest::blocking::Client;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub trait LlmClient: Send + Sync {
    fn chat(&self, model: &str, messages: &[ChatMessage], temperature: f32) -> Result<String>;
}

#[derive(Clone)]
pub struct OllamaClient {
    endpoint: String,
    http: Client,
}

#[derive(Clone)]
pub struct OpenAiCompatibleClient {
    endpoint: String,
    api_key: Option<String>,
    http: Client,
}

impl OllamaClient {
    pub fn new(endpoint: String, timeout_seconds: u64) -> Result<Self> {
        let http = Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_seconds))
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self { endpoint, http })
    }
}

impl OpenAiCompatibleClient {
    pub fn new(endpoint: String, api_key: Option<String>, timeout_seconds: u64) -> Result<Self> {
        let http = Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_seconds))
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self {
            endpoint,
            api_key,
            http,
        })
    }
}

impl LlmClient for OllamaClient {
    fn chat(&self, model: &str, messages: &[ChatMessage], temperature: f32) -> Result<String> {
        #[derive(Serialize)]
        struct Payload<'a> {
            model: &'a str,
            stream: bool,
            messages: &'a [ChatMessage],
            options: serde_json::Value,
        }

        #[derive(Deserialize)]
        struct Resp {
            message: Option<ChatMessage>,
        }

        let payload = Payload {
            model,
            stream: false,
            messages,
            options: serde_json::json!({"temperature": temperature}),
        };

        let resp = self
            .http
            .post(&self.endpoint)
            .header(CONTENT_TYPE, "application/json")
            .header(ACCEPT, "application/json")
            .json(&payload)
            .send()
            .with_context(|| format!("failed request to {}", self.endpoint))?;

        let status = resp.status();
        let body = resp.text().context("failed to read response body")?;
        if !status.is_success() {
            return Err(anyhow!("HTTP {} from {}: {}", status, self.endpoint, body));
        }

        let parsed: Resp = serde_json::from_str(&body)
            .with_context(|| format!("invalid JSON from {}: {}", self.endpoint, body))?;
        let content = parsed
            .message
            .map(|m| m.content)
            .ok_or_else(|| anyhow!("unexpected Ollama response shape: {body}"))?;
        Ok(content)
    }
}

impl LlmClient for OpenAiCompatibleClient {
    fn chat(&self, model: &str, messages: &[ChatMessage], temperature: f32) -> Result<String> {
        #[derive(Serialize)]
        struct Payload<'a> {
            model: &'a str,
            messages: &'a [ChatMessage],
            temperature: f32,
        }

        #[derive(Deserialize)]
        struct Choice {
            message: ChatMessage,
        }

        #[derive(Deserialize)]
        struct Resp {
            choices: Vec<Choice>,
        }

        let payload = Payload {
            model,
            messages,
            temperature,
        };

        let mut request = self
            .http
            .post(&self.endpoint)
            .header(CONTENT_TYPE, "application/json")
            .header(ACCEPT, "application/json");

        if let Some(api_key) = &self.api_key {
            request = request.header(AUTHORIZATION, format!("Bearer {api_key}"));
        }

        let resp = request
            .json(&payload)
            .send()
            .with_context(|| format!("failed request to {}", self.endpoint))?;

        let status = resp.status();
        let body = resp.text().context("failed to read response body")?;
        if !status.is_success() {
            return Err(anyhow!("HTTP {} from {}: {}", status, self.endpoint, body));
        }

        let parsed: Resp = serde_json::from_str(&body)
            .with_context(|| format!("invalid JSON from {}: {}", self.endpoint, body))?;
        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("unexpected OpenAI-compatible response shape: {body}"))?;
        Ok(choice.message.content)
    }
}
