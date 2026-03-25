use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct ModelRouting {
    pub text_model: String,
    pub planner_model: Option<String>,
    pub coder_model: Option<String>,
    pub image_model: Option<String>,
}

impl ModelRouting {
    pub fn model_for_role<'a>(&'a self, role: &str) -> &'a str {
        match role {
            "planner" => self
                .planner_model
                .as_deref()
                .unwrap_or(self.text_model.as_str()),
            "coder" => self
                .coder_model
                .as_deref()
                .unwrap_or(self.text_model.as_str()),
            "image" => self
                .image_model
                .as_deref()
                .unwrap_or(self.text_model.as_str()),
            _ => self.text_model.as_str(),
        }
    }

    pub fn as_json(&self) -> serde_json::Value {
        serde_json::json!({
            "text_model": self.text_model,
            "planner_model": self.planner_model,
            "coder_model": self.coder_model,
            "image_model": self.image_model,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub provider: String,
    pub endpoint: String,
    pub api_key: Option<String>,
    pub workspace: PathBuf,
    pub routing: ModelRouting,
    pub max_steps: u32,
    pub shell_timeout_seconds: u64,
    pub auto_mode: bool,
    pub temperature: f32,
    pub memory_file: Option<PathBuf>,
    pub plugins_dir: Option<PathBuf>,
    pub max_delegation_depth: u32,
}

impl Settings {
    pub fn workspace_resolved(&self) -> PathBuf {
        self.workspace
            .canonicalize()
            .unwrap_or_else(|_| self.workspace.clone())
    }

    pub fn memory_file_resolved(&self) -> PathBuf {
        let base = self.workspace_resolved();
        match &self.memory_file {
            Some(path) if path.is_absolute() => path.clone(),
            Some(path) => base.join(path),
            None => base.join(".local_codex").join("memory.json"),
        }
    }

    pub fn plugins_dir_resolved(&self) -> PathBuf {
        let base = self.workspace_resolved();
        match &self.plugins_dir {
            Some(path) if path.is_absolute() => path.clone(),
            Some(path) => base.join(path),
            None => base.join(".local_codex").join("plugins"),
        }
    }

    pub fn session_file_default(&self) -> PathBuf {
        self.workspace_resolved()
            .join(".local_codex")
            .join("session.json")
    }
}

pub fn normalize_path(base: &Path, value: Option<PathBuf>, fallback: PathBuf) -> PathBuf {
    match value {
        Some(path) if path.is_absolute() => path,
        Some(path) => base.join(path),
        None => fallback,
    }
}
