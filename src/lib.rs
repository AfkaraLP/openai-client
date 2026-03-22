//! # `OpenAI` API
//!
//! Interact with `OpenAI` compatible APIs easily, and build agents with function calling rather swiftly.
//!
//! ## Quickstart
//!
//! ```rust
//! use openai_client::prelude::*;
//!
//! pub fn main() {
//!     let client = OpenAIClient::new(
//!         "http://localhost:1234/v1",
//!         "Qwen3/Qwen3-0.8B",
//!         None,
//!     );
//!     client.get_completion(&[ChatCompletionMessageParam::new_system("hi")], &ToolMap::default());
//!     // do things with the completions now
//! }
#[cfg(all(feature = "runtime_agnostic", feature = "tokio_runtime"))]
compile_error!("You can only choose a single runtime");

pub mod prelude;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::{HashMap, hash_map::Values},
    pin::Pin,
};

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Request(reqwest::Error),
    Deserialization(serde_json::Error),
    Response,
}

pub type BoxedTool<'a> = Box<dyn ToolCallFn + Send + Sync + 'a>;
#[derive(Default)]
pub struct ToolMap<'a>(HashMap<&'static str, BoxedTool<'a>>);

impl<'a> ToolMap<'a> {
    #[must_use]
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    #[must_use]
    pub fn values(&self) -> Values<'_, &str, BoxedTool<'_>> {
        self.0.values()
    }

    #[must_use]
    /// Add a tool to the context of the client - uses the builder pattern.
    pub fn register_tool<T: ToolCallFn + Send + Sync + 'a>(mut self, tool: T) -> Self {
        self.0.insert(tool.get_name(), Box::new(tool));
        self
    }

    #[must_use]
    pub fn get(&self, key: &str) -> Option<&BoxedTool<'_>> {
        self.0.get(key)
    }
}

/// Client for interacting with an openai compatible API.
pub struct OpenAIClient {
    pub url: String,
    pub client: reqwest::Client,
    pub model: String,
    pub header_kv: Option<(String, String)>,
}

impl OpenAIClient {
    #[must_use]
    pub fn new(
        url: impl Into<String>,
        model: impl Into<String>,
        header_kv: Option<(String, String)>,
    ) -> Self {
        let client = reqwest::Client::new();
        Self {
            url: url.into(),
            client,
            model: model.into(),
            header_kv,
        }
    }

    #[must_use]
    pub fn set_key_pair(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.header_kv = Some((key.into(), value.into()));
        self
    }

    #[must_use]
    pub fn set_bearer_auth(mut self, token: impl Into<String>) -> Self {
        self.header_kv = Some(("Authorization".into(), token.into()));
        self
    }

    /// Get a response that fits a certain schema.
    ///
    /// # Usage
    ///
    /// ```rust
    /// use openai_client::prelude::*;
    /// use serde::{Serialize, Deserialize};
    ///
    /// #[derive(JsonSchema, Serialize, Deserialize)]
    /// struct MyStruct {
    ///     thing: String
    /// }
    ///
    /// let client = OpenAIClient::new(
    ///     "http://localhost:1234/v1",
    ///     "Qwen3/Qwen3-0.8B",
    ///     None,
    /// );
    ///
    /// client.get_structured_response::<MyStruct>(&[ChatCompletionMessageParam::new_user("this is a test")]);
    /// ```
    ///
    /// # Errors
    ///
    /// Can fail on API rejects or if the model does not respond with the correct struct
    pub async fn get_structured_response<T: DeserializeOwned + JsonSchema>(
        &self,
        messages: &[ChatCompletionMessageParam],
    ) -> Result<T> {
        let schema = schemars::schema_for!(T);
        let req = ChatCompletionRequest {
            messages: messages.to_vec(),
            model: self.model.clone(),
            temperature: Some(6.7),
            tools: None,
            response_format: Some(ResponseFormat {
                kind: "json_schema".to_string(),
                json_schema: JsonSchemaConfig {
                    name: std::any::type_name::<T>()
                        .rsplit("::")
                        .next()
                        .unwrap_or("response")
                        .to_string(),
                    schema: serde_json::to_value(schema).map_err(Error::Deserialization)?,
                    strict: true,
                },
            }),
        };
        let mut reqbuilder = self
            .client
            .post(format!("{}/chat/completions", self.url))
            .json(&req);
        if let Some((key, token)) = &self.header_kv {
            reqbuilder = reqbuilder.bearer_auth(token).header(key, token);
        }
        let res_str = reqbuilder
            .send()
            .await
            .map_err(Error::Request)?
            .text()
            .await
            .map_err(Error::Request)?;
        let response: ChatCompletionResponse =
            serde_json::from_str(&res_str).map_err(Error::Deserialization)?;
        let content = response
            .first()
            .and_then(|v| v.content.as_deref())
            .ok_or(Error::Response)?;
        serde_json::from_str(content).map_err(Error::Deserialization)
    }

    /// Continue the conversation with a given set of tools.
    ///
    /// # Errors
    ///
    /// ## The API has some error
    ///
    /// this can include:
    /// - `Rate Limit` / `Unauthorized` (403)
    /// - Server Errors (idk message openai in that case or sumn)
    ///
    /// ## Deserialization
    ///
    /// if the API response is not in the standard openai format then this will also fail.
    pub async fn get_completion<'a>(
        &'a self,
        messages: &[ChatCompletionMessageParam],
        tools: &ToolMap<'a>,
    ) -> Result<ChatCompletionResponse> {
        let req_tools: Vec<serde_json::Value> =
            tools.values().map(|tool| tool.to_json_value()).collect();
        let request = ChatCompletionRequest {
            model: self.model.clone(),
            tools: if req_tools.is_empty() {
                None
            } else {
                Some(req_tools)
            },
            messages: messages.to_vec(),
            temperature: Some(0.67),
            response_format: None,
        };
        let mut reqbuilder = self
            .client
            .post(format!("{}/chat/completions", self.url))
            .json(&request);
        if let Some((key, token)) = &self.header_kv {
            reqbuilder = reqbuilder.bearer_auth(token).header(key, token);
        }
        let res_str = reqbuilder
            .send()
            .await
            .map_err(Error::Request)?
            .text()
            .await
            .map_err(Error::Request)?;
        serde_json::from_str(&res_str).map_err(Error::Deserialization)
    }

    /// Run the LLM in a loop until it stops calling tools and just responds.
    ///
    /// # Errors
    ///
    /// - There was no completion.
    /// - The completion did not have any messages.
    pub async fn run_agent(
        &self,
        system_prompt: impl Into<String>,
        prompt: impl Into<String>,
        tools: &ToolMap<'_>,
    ) -> Result<String> {
        let system_prompt = ChatCompletionMessageParam::new_system(system_prompt.into());
        let user_prompt = ChatCompletionMessageParam::new_user(prompt.into());
        let mut prompts = vec![system_prompt, user_prompt];
        loop {
            let completion: ChatCompletionResponse = self.get_completion(&prompts, tools).await?;
            let response = completion.first().ok_or(Error::Response)?;

            if response.has_tools() {
                prompts.push(ChatCompletionMessageParam::new_assistant_with_tools(
                    response.content.as_deref().unwrap_or(""),
                    Some(
                        response
                            .tool_calls
                            .iter()
                            .map(|tc| ToolCall {
                                id: tc.id.clone(),
                                call_type: tc.ftype.clone(),
                                function: ToolCallFunction {
                                    name: tc.function.name.clone(),
                                    arguments: tc.function.arguments.clone(),
                                },
                            })
                            .collect(),
                    ),
                ));

                let tool_responses = response.call_tools(tools).await;
                for tool_response in tool_responses {
                    prompts.push(tool_response);
                }
            } else {
                if let Some(assistant_message) = &response.content
                    && !assistant_message.is_empty()
                {
                    prompts.push(ChatCompletionMessageParam::new_assistant(assistant_message));
                }
                break Ok(response
                    .content
                    .clone()
                    .unwrap_or_else(|| String::from("Agent did not respond on last turn.")));
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatCompletionMessageParam>,
    pub model: String,
    pub temperature: Option<f32>,
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    kind: String,
    json_schema: JsonSchemaConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JsonSchemaConfig {
    name: String,
    schema: serde_json::Value,
    strict: bool,
}

impl ChatCompletionRequest {
    #[must_use]
    pub const fn new(
        messages: Vec<ChatCompletionMessageParam>,
        model: String,
        temperature: Option<f32>,
        tools: Option<Vec<serde_json::Value>>,
        response_format: Option<ResponseFormat>,
    ) -> Self {
        Self {
            messages,
            model,
            temperature,
            tools,
            response_format,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionMessageParam {
    pub content: String,
    pub role: AIChatRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatCompletionMessageParam {
    #[must_use]
    pub fn new<S: Into<String>>(content: S, role: AIChatRole) -> Self {
        Self {
            content: content.into(),
            role,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[must_use]
    pub fn new_user<S: Into<String>>(content: S) -> Self {
        Self::new(content, AIChatRole::User)
    }

    #[must_use]
    pub fn new_assistant<S: Into<String>>(content: S) -> Self {
        Self::new(content, AIChatRole::Assistant)
    }

    #[must_use]
    pub fn new_assistant_with_tools<S: Into<String>>(
        content: S,
        tool_calls: Option<Vec<ToolCall>>,
    ) -> Self {
        Self {
            content: content.into(),
            role: AIChatRole::Assistant,
            tool_calls,
            tool_call_id: None,
        }
    }

    #[must_use]
    pub fn new_tool<S: Into<String>>(content: S, tool_call_id: &str) -> Self {
        Self {
            content: content.into(),
            role: AIChatRole::Tool,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.to_string()),
        }
    }

    #[must_use]
    pub fn new_system<S: Into<String>>(content: S) -> Self {
        Self::new(content, AIChatRole::System)
    }
}

impl Default for ChatCompletionMessageParam {
    fn default() -> Self {
        Self {
            content: String::new(),
            role: AIChatRole::User,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

/// Utility method to generate a quick start of LLM turn.
pub fn new_system_user_turn(
    system_prompt: impl Into<String>,
    user_prompt: impl Into<String>,
) -> Vec<ChatCompletionMessageParam> {
    vec![
        ChatCompletionMessageParam::new_system(system_prompt),
        ChatCompletionMessageParam::new_user(user_prompt),
    ]
}

/// Utility method to generate a quick user completion message parameter without developing carpal tunnel.
pub fn user_message(content: impl Into<String>) -> ChatCompletionMessageParam {
    ChatCompletionMessageParam::new_user(content)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AIChatRole {
    #[serde(rename = "user")]
    User,
    #[serde(rename = "tool")]
    Tool,
    #[serde(rename = "system")]
    System,
    #[serde(rename = "assistant")]
    Assistant,
    Custom(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub choices: Vec<ChatCompletionResponseChoice>,
}

impl ChatCompletionResponse {
    #[must_use]
    /// Get the first response from the model.
    pub fn first(&self) -> Option<&ChatCompletionResponseMessage> {
        self.choices.first().map(|v| &v.message)
    }

    #[must_use]
    /// This method is currently stale as this is only usable when we configure the model to send more than a single response.
    pub fn get(&self, n: usize) -> Option<&ChatCompletionResponseMessage> {
        self.choices.get(n).map(|v| &v.message)
    }
}

impl From<Vec<ChatCompletionResponseChoice>> for ChatCompletionResponse {
    fn from(choices: Vec<ChatCompletionResponseChoice>) -> Self {
        Self { choices }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponseChoice {
    pub message: ChatCompletionResponseMessage,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponseMessage {
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub role: AIChatRole,
    pub tool_calls: Vec<ToolInvocation>,
}

impl ChatCompletionResponseMessage {
    #[must_use]
    #[inline]
    pub fn has_tools(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    #[must_use]
    pub async fn call_tools(&self, tools: &ToolMap<'_>) -> Vec<ChatCompletionMessageParam> {
        let mut responses = Vec::new();
        for requested_tool in &self.tool_calls {
            let result = requested_tool.call(tools).await;
            responses.push(ChatCompletionMessageParam::new_tool(
                result,
                &requested_tool.id,
            ));
        }
        responses
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolInvocation {
    pub id: String,
    #[serde(rename = "type")]
    pub ftype: String,
    pub function: ToolInvocationFunction,
}

pub async fn sleep(duration: std::time::Duration) {
    #[cfg(feature = "runtime_agnostic")]
    futures_timer::Delay::new(duration).await;
    #[cfg(feature = "tokio_runtime")]
    tokio::time::sleep(duration).await;
}

impl ToolInvocation {
    pub async fn call(&self, tools: &ToolMap<'_>) -> String {
        let Some(tool) = tools.get(&self.function.name) else {
            let name = self.function.name.as_str();
            return format!("{name} tool is unknown.");
        };
        sleep(tool.get_timeout_wait()).await;
        tool.invoke(
            &serde_json::from_str(&self.function.arguments).unwrap_or(serde_json::Value::Null),
        )
        .await
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolInvocationFunction {
    pub arguments: String,
    pub name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

impl std::fmt::Display for ChatCompletionMessageParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "[{:#?}: {}]", self.role, self.content)
    }
}

pub trait IntoPinBox<'a>: Sized + Send + ToString + 'a {
    /// Allow something to be returned as a `Pin<Box<dyn Future<Output = String>>>`
    fn into_pin_box(self) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
        Box::pin(async move { self.to_string() })
    }
}

impl<'a, T> IntoPinBox<'a> for T where T: Sized + Send + ToString + 'a {}

/// Turn anything you want into a callable tool.
///
/// # Usage
///
/// ```rust
/// use openai_client::prelude::*;
/// pub struct MyTool {
///     some_context: String,
/// }
/// pub async fn foo(a: String, b: String) -> String {
///     reqwest::get(format!("{a}/{b}"))
///         .await
///         .unwrap()
///         .text()
///         .await
///         .unwrap()
/// }
///
/// impl ToolCallFn for MyTool {
///     fn get_name(&self) -> &'static str {
///         "my_tool"
///     }
///
///     fn get_description(&self) -> &'static str {
///         "this tool does awesome things"
///     }
///
///     fn invoke(&self, args: &serde_json::Value) -> std::pin::Pin<Box<dyn Future<Output = String> + Send>> {
///         let Some(serde_json::Value::String(cool_arg)) = args.get("cool_arg") else {
///             return "failed".into_pin_box();
///         };
///         let context = self.some_context.clone();
///         let cool_arg = cool_arg.clone();
///         Box::pin(async move { foo(cool_arg, context).await })
///     }
///
///     fn get_args(&self) -> Vec<ToolCallArgDescriptor> {
///         vec![ToolCallArgDescriptor::new(
///             true,
///             "cool_arg",
///             ToolCallArgType::String,
///             "this is a really cool argument that the llm knows about now",
///         )]
///     }
///
///     fn get_timeout_wait(&self) -> std::time::Duration {
///          std::time::Duration::from_millis(50)
///     }
/// }
///
/// ToolMap::new().register_tool(MyTool {
///     some_context: String::from("cool tool stuff"),
/// });
/// ```
///
pub trait ToolCallFn {
    #[must_use]
    fn get_name(&self) -> &'static str;
    #[must_use]
    fn get_description(&self) -> &'static str;

    fn invoke<'a>(
        &'a self,
        args: &'a serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = String> + Send + 'a>>;

    #[must_use]
    fn to_json_value(&self) -> serde_json::Value {
        let name = self.get_name();
        let description = self.get_description();
        let args = self.get_args();
        let required = args
            .iter()
            .filter_map(|arg| if arg.required { Some(arg.name) } else { None })
            .collect::<Vec<&str>>();
        let props: HashMap<String, serde_json::Value> =
            args.iter().map(ToolCallArgDescriptor::serialize).collect();
        let json = serde_json::json!(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required
                    }
                }
            }
        );
        json
    }

    #[must_use]
    fn get_args(&self) -> Vec<ToolCallArgDescriptor>;

    #[must_use]
    fn get_timeout_wait(&self) -> std::time::Duration {
        std::time::Duration::from_millis(3000)
    }
}

#[derive(Debug, Clone, Default)]
pub struct ToolCallArgDescriptor {
    pub required: bool,
    pub name: &'static str,
    pub argtype: ToolCallArgType,
    pub description: &'static str,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub enum ToolCallArgType {
    #[default]
    #[serde(rename = "string")]
    String,
    #[serde(rename = "number")]
    Number,
    #[serde(rename = "bool")]
    Bool,
}

impl ToolCallArgDescriptor {
    #[must_use]
    pub fn new(
        required: bool,
        name: &'static str,
        argtype: ToolCallArgType,
        description: &'static str,
    ) -> Self {
        Self {
            required,
            name,
            argtype,
            description,
        }
    }

    #[must_use]
    pub fn string(name: &'static str, description: &'static str) -> Self {
        Self::required(name, ToolCallArgType::String, description)
    }

    #[must_use]
    pub fn number(name: &'static str, description: &'static str) -> Self {
        Self::required(name, ToolCallArgType::Number, description)
    }

    #[must_use]
    pub fn bool(name: &'static str, description: &'static str) -> Self {
        Self::required(name, ToolCallArgType::Bool, description)
    }

    #[must_use]
    pub fn required(
        name: &'static str,
        argtype: ToolCallArgType,
        description: &'static str,
    ) -> Self {
        Self::new(true, name, argtype, description)
    }

    #[must_use]
    pub fn optional(
        name: &'static str,
        argtype: ToolCallArgType,
        description: &'static str,
    ) -> Self {
        Self::new(false, name, argtype, description)
    }

    #[must_use]
    fn serialize(&self) -> (String, serde_json::Value) {
        (
            self.name.to_string(),
            serde_json::json!({
                "type": self.argtype,
                "description": self.description
            }),
        )
    }
}
