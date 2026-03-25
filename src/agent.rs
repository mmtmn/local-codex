use crate::config::ModelRouting;
use crate::llm::{ChatMessage, LlmClient};
use crate::tools::ToolExecutor;
use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

const TOOL_RESULT_PROMPT: &str =
    "Tool execution result:\n{result}\n\nContinue. If the task is complete, respond with a final message.";
const PARALLEL_RESULT_PROMPT: &str =
    "Parallel tool execution result:\n{result}\n\nContinue. If the task is complete, respond with a final message.";
const DELEGATE_RESULT_PROMPT: &str =
    "Delegated subtask result:\n{result}\n\nContinue. If the task is complete, respond with a final message.";

#[derive(Clone, Debug)]
struct AgentSession {
    messages: Vec<ChatMessage>,
    tool_history: Vec<Value>,
    delegate_history: Vec<Value>,
}

#[derive(Clone, Debug)]
struct Action {
    action_type: String,
    payload: Map<String, Value>,
}

#[derive(Clone, Debug)]
pub struct AgentSnapshot {
    pub name: String,
    pub messages: Vec<ChatMessage>,
    pub tool_history: Vec<Value>,
    pub delegate_history: Vec<Value>,
}

pub struct Orchestrator {
    llm: Box<dyn LlmClient>,
    tools: ToolExecutor,
    workspace_root: String,
    max_steps: u32,
    temperature: f32,
    max_delegation_depth: u32,
    routing: ModelRouting,
    agents: HashMap<String, AgentSession>,
    active_agent: String,
    delegate_counter: u64,
    show_tool_events: bool,
}

impl Orchestrator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        llm: Box<dyn LlmClient>,
        tools: ToolExecutor,
        workspace_root: String,
        max_steps: u32,
        temperature: f32,
        max_delegation_depth: u32,
        routing: ModelRouting,
        show_tool_events: bool,
    ) -> Self {
        let mut this = Self {
            llm,
            tools,
            workspace_root,
            max_steps: max_steps.max(1),
            temperature,
            max_delegation_depth,
            routing,
            agents: HashMap::new(),
            active_agent: "main".to_string(),
            delegate_counter: 0,
            show_tool_events,
        };
        this.ensure_agent("main");
        this
    }

    pub fn active_agent(&self) -> &str {
        &self.active_agent
    }

    pub fn model_routing(&self) -> Value {
        self.routing.as_json()
    }

    pub fn list_agents(&self) -> Vec<String> {
        let mut names = self.agents.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }

    pub fn has_agent(&self, name: &str) -> bool {
        self.agents.contains_key(name)
    }

    pub fn create_agent(&mut self, name: &str) -> Result<()> {
        let normalized = normalize_agent_name(name)?;
        self.ensure_agent(&normalized);
        Ok(())
    }

    pub fn set_active_agent(&mut self, name: &str) -> Result<()> {
        let normalized = normalize_agent_name(name)?;
        self.ensure_agent(&normalized);
        self.active_agent = normalized;
        Ok(())
    }

    pub fn reset_active(&mut self) -> Result<()> {
        let active = self.active_agent.clone();
        self.reset_agent(&active)
    }

    pub fn reset_agent(&mut self, name: &str) -> Result<()> {
        let normalized = normalize_agent_name(name)?;
        self.ensure_agent(&normalized);
        let fresh = self.new_session(&normalized);
        self.agents.insert(normalized, fresh);
        Ok(())
    }

    pub fn run_active(&mut self, user_prompt: &str) -> String {
        let active = self.active_agent.clone();
        self.run_with_depth(&active, user_prompt, 0, self.max_steps)
    }

    pub fn run_with_agent(&mut self, name: &str, user_prompt: &str) -> Result<String> {
        let normalized = normalize_agent_name(name)?;
        self.ensure_agent(&normalized);
        Ok(self.run_with_depth(&normalized, user_prompt, 0, self.max_steps))
    }

    pub fn agent_snapshot(&self, name: &str) -> Option<AgentSnapshot> {
        self.agents.get(name).map(|session| AgentSnapshot {
            name: name.to_string(),
            messages: session.messages.clone(),
            tool_history: session.tool_history.clone(),
            delegate_history: session.delegate_history.clone(),
        })
    }

    pub fn reload_plugins(&mut self) -> String {
        let result = self
            .tools
            .execute("reload_plugins", &json!({}), &self.routing.as_json());
        self.refresh_all_system_prompts();
        result
    }

    pub fn session_payload(&self) -> Value {
        let mut agents_obj = Map::new();
        for name in self.list_agents() {
            if let Some(session) = self.agents.get(&name) {
                agents_obj.insert(
                    name,
                    json!({
                        "messages": session.messages,
                        "tool_history": session.tool_history,
                        "delegate_history": session.delegate_history,
                    }),
                );
            }
        }

        json!({
            "version": 3,
            "active_agent": self.active_agent,
            "delegate_counter": self.delegate_counter,
            "agents": agents_obj,
        })
    }

    pub fn save_session<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        let target = path.as_ref().to_path_buf();
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed creating directory {}", parent.display()))?;
        }

        let payload = self.session_payload();
        let text = serde_json::to_string_pretty(&payload).context("failed serializing session")?;
        fs::write(&target, text)
            .with_context(|| format!("failed writing session {}", target.display()))?;
        Ok(target)
    }

    pub fn load_session<P: AsRef<Path>>(&mut self, path: P) -> Result<PathBuf> {
        let source = path.as_ref().to_path_buf();
        let text = fs::read_to_string(&source)
            .with_context(|| format!("failed reading session {}", source.display()))?;
        let payload: Value = serde_json::from_str(&text)
            .with_context(|| format!("invalid JSON in session {}", source.display()))?;

        let mut loaded = self.restore_agents_from_payload(&payload);
        if loaded.is_empty() {
            loaded.insert("main".to_string(), self.new_session("main"));
        }
        self.agents = loaded;

        let active = payload
            .get("active_agent")
            .and_then(Value::as_str)
            .unwrap_or("main");

        if is_valid_agent_name(active) && self.agents.contains_key(active) {
            self.active_agent = active.to_string();
        } else if self.agents.contains_key("main") {
            self.active_agent = "main".to_string();
        } else {
            let mut names = self.list_agents();
            names.sort();
            self.active_agent = names.first().cloned().unwrap_or_else(|| "main".to_string());
        }

        self.delegate_counter = payload
            .get("delegate_counter")
            .and_then(Value::as_u64)
            .unwrap_or(0);

        Ok(source)
    }

    fn restore_agents_from_payload(&self, payload: &Value) -> HashMap<String, AgentSession> {
        let mut loaded = HashMap::new();

        if let Some(agents_obj) = payload.get("agents").and_then(Value::as_object) {
            for (name, raw) in agents_obj {
                if !is_valid_agent_name(name) {
                    continue;
                }
                if let Some(raw_obj) = raw.as_object() {
                    loaded.insert(name.clone(), self.restore_single_agent(name, raw_obj));
                }
            }
            return loaded;
        }

        // Backward compatibility with single-agent sessions.
        if let Some(messages) = payload.get("messages") {
            if let Some(raw_messages) = messages.as_array() {
                let session = AgentSession {
                    messages: self.restore_messages("main", raw_messages),
                    tool_history: payload
                        .get("tool_history")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default(),
                    delegate_history: payload
                        .get("delegate_history")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default(),
                };
                loaded.insert("main".to_string(), session);
            }
        }

        loaded
    }

    fn restore_single_agent(&self, name: &str, raw: &Map<String, Value>) -> AgentSession {
        let messages = raw
            .get("messages")
            .and_then(Value::as_array)
            .map(|entries| self.restore_messages(name, entries))
            .unwrap_or_else(|| {
                vec![ChatMessage {
                    role: "system".to_string(),
                    content: self.system_prompt_for(name),
                }]
            });

        let tool_history = raw
            .get("tool_history")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let delegate_history = raw
            .get("delegate_history")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        AgentSession {
            messages,
            tool_history,
            delegate_history,
        }
    }

    fn restore_messages(&self, name: &str, raw_messages: &[Value]) -> Vec<ChatMessage> {
        let mut restored = Vec::new();
        for raw in raw_messages {
            let Some(obj) = raw.as_object() else {
                continue;
            };
            let Some(role) = obj.get("role").and_then(Value::as_str) else {
                continue;
            };
            let Some(content) = obj.get("content").and_then(Value::as_str) else {
                continue;
            };
            if role == "system" {
                continue;
            }
            restored.push(ChatMessage {
                role: role.to_string(),
                content: content.to_string(),
            });
        }

        let mut messages = vec![ChatMessage {
            role: "system".to_string(),
            content: self.system_prompt_for(name),
        }];
        messages.extend(restored);
        messages
    }

    fn ensure_agent(&mut self, name: &str) {
        if self.agents.contains_key(name) {
            return;
        }
        self.agents.insert(name.to_string(), self.new_session(name));
    }

    fn new_session(&self, name: &str) -> AgentSession {
        AgentSession {
            messages: vec![ChatMessage {
                role: "system".to_string(),
                content: self.system_prompt_for(name),
            }],
            tool_history: Vec::new(),
            delegate_history: Vec::new(),
        }
    }

    fn refresh_all_system_prompts(&mut self) {
        let names = self.list_agents();
        let mut prompts = HashMap::new();
        for name in &names {
            prompts.insert(name.clone(), self.system_prompt_for(name));
        }

        for name in names {
            if let Some(session) = self.agents.get_mut(&name) {
                let non_system = session
                    .messages
                    .iter()
                    .filter(|message| message.role != "system")
                    .cloned()
                    .collect::<Vec<_>>();

                let mut refreshed = Vec::with_capacity(non_system.len() + 1);
                refreshed.push(ChatMessage {
                    role: "system".to_string(),
                    content: prompts.remove(&name).unwrap_or_default(),
                });
                refreshed.extend(non_system);
                session.messages = refreshed;
            }
        }
    }

    fn system_prompt_for(&self, agent_name: &str) -> String {
        let tool_specs = self.tools.render_specs_for_prompt();
        format!(
            "You are LocalCodex, a terminal coding assistant.\n\nYou can use tools to inspect and modify files, run shell commands, and orchestrate sub-agents.\nOnly use tools when needed.\n\nWorkspace root:\n{}\n\nCurrent agent name:\n{}\n\nAvailable tools JSON:\n{}\n\nResponse rules:\n1) Always respond with exactly one JSON object and no surrounding markdown.\n2) Use one of these shapes:\n   - {{\"type\": \"tool\", \"tool\": \"<tool_name>\", \"args\": {{...}}, \"reason\": \"short reason\"}}\n   - {{\"type\": \"parallel_tools\", \"calls\": [{{\"tool\": \"<tool>\", \"args\": {{...}}}}], \"max_workers\": 4, \"reason\": \"short reason\"}}\n   - {{\"type\": \"delegate\", \"prompt\": \"subtask prompt\", \"agent_name\": \"optional_named_agent\", \"max_steps\": 6, \"reason\": \"short reason\"}}\n   - {{\"type\": \"final\", \"message\": \"final user-facing response\"}}\n3) If a tool errors, adapt and continue.\n4) Prefer minimal, safe changes that satisfy the request.\n5) Keep delegate depth <= {}.\n",
            self.workspace_root, agent_name, tool_specs, self.max_delegation_depth,
        )
    }

    fn run_with_depth(
        &mut self,
        agent_name: &str,
        user_prompt: &str,
        depth: u32,
        max_steps: u32,
    ) -> String {
        self.ensure_agent(agent_name);
        self.append_message(
            agent_name,
            ChatMessage {
                role: "user".to_string(),
                content: user_prompt.to_string(),
            },
        );

        let step_limit = max_steps.max(1);

        for step in 0..step_limit {
            let role = if step == 0 {
                if depth == 0 {
                    "planner"
                } else {
                    "coder"
                }
            } else {
                "coder"
            };
            let model = self.routing.model_for_role(role).to_string();

            let messages = self
                .agents
                .get(agent_name)
                .map(|session| session.messages.clone())
                .unwrap_or_default();

            let model_response = match self.llm.chat(&model, &messages, self.temperature) {
                Ok(value) => value,
                Err(err) => {
                    return format!("I hit an LLM error while using model '{model}': {err}");
                }
            };

            let action = Self::parse_action(&model_response);
            let Some(action) = action else {
                self.append_message(
                    agent_name,
                    ChatMessage {
                        role: "assistant".to_string(),
                        content: model_response.clone(),
                    },
                );
                return model_response.trim().to_string();
            };

            let payload_text =
                serde_json::to_string(&action.payload).unwrap_or_else(|_| "{}".to_string());
            self.append_message(
                agent_name,
                ChatMessage {
                    role: "assistant".to_string(),
                    content: payload_text,
                },
            );

            match action.action_type.as_str() {
                "final" => {
                    let message = action
                        .payload
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if message.is_empty() {
                        return "I do not have a final message yet.".to_string();
                    }
                    return message;
                }
                "tool" => {
                    let result = self.handle_tool_action(agent_name, &action.payload);
                    self.append_message(
                        agent_name,
                        ChatMessage {
                            role: "user".to_string(),
                            content: TOOL_RESULT_PROMPT.replace("{result}", &result),
                        },
                    );
                }
                "parallel_tools" => {
                    let result = self.handle_parallel_action(agent_name, &action.payload);
                    self.append_message(
                        agent_name,
                        ChatMessage {
                            role: "user".to_string(),
                            content: PARALLEL_RESULT_PROMPT.replace("{result}", &result),
                        },
                    );
                }
                "delegate" => {
                    let result =
                        self.handle_delegate_action(agent_name, &action.payload, depth, step_limit);
                    self.append_message(
                        agent_name,
                        ChatMessage {
                            role: "user".to_string(),
                            content: DELEGATE_RESULT_PROMPT.replace("{result}", &result),
                        },
                    );
                }
                _ => {
                    self.append_message(agent_name, ChatMessage {
                        role: "user".to_string(),
                        content: "Invalid action type. Respond with JSON using type='tool', type='parallel_tools', type='delegate', or type='final'.".to_string(),
                    });
                }
            }
        }

        "I hit the max tool step limit before completing the task. Please retry with a narrower request or higher --max-steps.".to_string()
    }

    fn handle_tool_action(&mut self, agent_name: &str, payload: &Map<String, Value>) -> String {
        let tool_name = payload
            .get("tool")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if tool_name.is_empty() {
            return "ERROR: tool action requires non-empty tool".to_string();
        }

        let args = payload
            .get("args")
            .cloned()
            .filter(|value| value.is_object())
            .unwrap_or_else(|| json!({}));

        self.emit_tool_event(&format!(
            "{} tool {} {}",
            agent_name,
            tool_name,
            serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string())
        ));

        let result = self
            .tools
            .execute(tool_name, &args, &self.routing.as_json());
        self.push_tool_history(
            agent_name,
            json!({
                "tool": tool_name,
                "args": args,
                "result": result,
            }),
        );

        self.emit_tool_event(&format!("{} result {}", agent_name, preview(&result, 240)));

        result
    }

    fn handle_parallel_action(&mut self, agent_name: &str, payload: &Map<String, Value>) -> String {
        let calls = payload.get("calls").cloned().unwrap_or_else(|| json!([]));
        let max_workers = payload
            .get("max_workers")
            .cloned()
            .unwrap_or_else(|| json!(4));

        let args = json!({
            "calls": calls,
            "max_workers": max_workers,
        });

        self.emit_tool_event(&format!(
            "{} parallel_tools {}",
            agent_name,
            serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string())
        ));

        let result = self
            .tools
            .execute("parallel_tools", &args, &self.routing.as_json());

        self.push_tool_history(
            agent_name,
            json!({
                "tool": "parallel_tools",
                "args": args,
                "result": result,
            }),
        );

        self.emit_tool_event(&format!("{} result {}", agent_name, preview(&result, 240)));

        result
    }

    fn handle_delegate_action(
        &mut self,
        parent_agent: &str,
        payload: &Map<String, Value>,
        depth: u32,
        parent_steps: u32,
    ) -> String {
        let prompt = payload
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if prompt.is_empty() {
            return "ERROR: delegate action requires non-empty prompt".to_string();
        }

        if depth >= self.max_delegation_depth {
            return format!(
                "ERROR: delegation depth limit reached ({}). Continue without delegating.",
                self.max_delegation_depth
            );
        }

        let default_steps = (parent_steps / 2).max(2);
        let child_steps = payload
            .get("max_steps")
            .and_then(Value::as_u64)
            .map(|value| value as u32)
            .unwrap_or(default_steps)
            .clamp(1, parent_steps.max(1));

        let child_name = match payload.get("agent_name").and_then(Value::as_str) {
            Some(value) if !value.trim().is_empty() => match normalize_agent_name(value) {
                Ok(name) => name,
                Err(err) => return format!("ERROR: invalid delegate agent_name: {err}"),
            },
            _ => self.next_delegate_name(depth + 1),
        };

        self.ensure_agent(&child_name);
        self.emit_tool_event(&format!(
            "{} delegate {} steps={} prompt={}",
            parent_agent,
            child_name,
            child_steps,
            preview(prompt, 120)
        ));

        let result = self.run_with_depth(&child_name, prompt, depth + 1, child_steps);

        self.push_delegate_history(
            parent_agent,
            json!({
                "agent_name": child_name,
                "prompt": prompt,
                "max_steps": child_steps,
                "result": result,
            }),
        );

        self.emit_tool_event(&format!(
            "{} delegate_result {}",
            parent_agent,
            preview(&result, 240)
        ));

        result
    }

    fn next_delegate_name(&mut self, depth: u32) -> String {
        loop {
            self.delegate_counter += 1;
            let candidate = format!("delegate_{}_{}", depth, self.delegate_counter);
            if !self.agents.contains_key(&candidate) {
                return candidate;
            }
        }
    }

    fn append_message(&mut self, agent_name: &str, message: ChatMessage) {
        if let Some(session) = self.agents.get_mut(agent_name) {
            session.messages.push(message);
        }
    }

    fn push_tool_history(&mut self, agent_name: &str, entry: Value) {
        if let Some(session) = self.agents.get_mut(agent_name) {
            session.tool_history.push(entry);
        }
    }

    fn push_delegate_history(&mut self, agent_name: &str, entry: Value) {
        if let Some(session) = self.agents.get_mut(agent_name) {
            session.delegate_history.push(entry);
        }
    }

    fn emit_tool_event(&self, message: &str) {
        if self.show_tool_events {
            println!("[tool] {message}");
        }
    }

    fn parse_action(text: &str) -> Option<Action> {
        let obj = Self::extract_json_object(text)?;
        let payload = obj.as_object()?.clone();
        let action_type = payload.get("type")?.as_str()?.to_string();
        Some(Action {
            action_type,
            payload,
        })
    }

    fn extract_json_object(text: &str) -> Option<Value> {
        let stripped = text.trim();
        if stripped.is_empty() {
            return None;
        }

        if let Some(value) = Self::try_parse_json_object(stripped) {
            return Some(value);
        }

        if stripped.starts_with("```") {
            let mut lines = stripped.lines();
            let _ = lines.next();
            let mut body = lines.collect::<Vec<_>>().join("\n");
            if let Some(pos) = body.rfind("```") {
                body.truncate(pos);
            }
            if let Some(value) = Self::try_parse_json_object(body.trim()) {
                return Some(value);
            }
        }

        for (idx, ch) in stripped.char_indices() {
            if ch != '{' {
                continue;
            }
            if let Some(value) = Self::try_parse_json_object(&stripped[idx..]) {
                return Some(value);
            }
        }

        None
    }

    fn try_parse_json_object(candidate: &str) -> Option<Value> {
        let mut deserializer = serde_json::Deserializer::from_str(candidate);
        let value = Value::deserialize(&mut deserializer).ok()?;
        if value.is_object() {
            Some(value)
        } else {
            None
        }
    }
}

fn preview(value: &str, max_chars: usize) -> String {
    let collapsed = value.replace('\n', " ");
    let mut chars = collapsed.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return collapsed;
    }
    chars.truncate(max_chars);
    let mut result = chars.into_iter().collect::<String>();
    result.push_str("...");
    result
}

fn normalize_agent_name(name: &str) -> Result<String> {
    let value = name.trim();
    if !is_valid_agent_name(value) {
        return Err(anyhow!(
            "agent name must match [A-Za-z0-9_-] and be 1-64 chars"
        ));
    }
    Ok(value.to_string())
}

fn is_valid_agent_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes.len() > 64 {
        return false;
    }
    bytes
        .iter()
        .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'_' || *byte == b'-')
}
