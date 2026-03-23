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
#![expect(
    clippy::implicit_return,
    clippy::question_mark_used,
    clippy::arbitrary_source_item_ordering,
    clippy::pattern_type_mismatch,
    reason = "Ugly style"
)]
#[cfg(all(feature = "runtime_agnostic", feature = "tokio_runtime"))]
compile_error!("You can only choose a single runtime");

pub mod prelude;

use core::{
    any::type_name,
    fmt::{self, Formatter},
    pin::Pin,
    result,
    time::Duration,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::{HashMap, hash_map::Values};
#[cfg(feature = "tokio_runtime")]
use tokio::time::sleep;

pub type Result<T> = result::Result<T, Error>;

#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    Deserialization(serde_json::Error),
    Request(reqwest::Error),
    Response,
}

pub type BoxedTool<'tools> = Box<dyn ToolCallFn + Send + Sync + 'tools>;
#[non_exhaustive]
#[derive(Default)]
pub struct ToolMap<'tool>(HashMap<&'static str, BoxedTool<'tool>>);

impl<'tool> ToolMap<'tool> {
    #[inline]
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&BoxedTool<'_>> {
        self.0.get(key)
    }

    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    #[inline]
    #[must_use]
    /// Add a tool to the context of the client - uses the builder pattern.
    pub fn register_tool<T: ToolCallFn + Send + Sync + 'tool>(mut self, tool: T) -> Self {
        self.0.insert(tool.get_name(), Box::new(tool));
        self
    }

    #[inline]
    #[must_use]
    pub fn values(&self) -> Values<'_, &str, BoxedTool<'_>> {
        self.0.values()
    }
}

#[non_exhaustive]
/// Client for interacting with an openai compatible API.
pub struct OpenAIClient {
    pub client: reqwest::Client,
    pub header_kv: Option<(String, String)>,
    pub model: String,
    pub url: String,
}

impl OpenAIClient {
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
    #[inline]
    pub async fn get_completion<'tool>(
        &'tool self,
        messages: &[ChatCompletionMessageParam],
        tools: &ToolMap<'tool>,
    ) -> Result<ChatCompletionResponse> {
        let req_tools: Vec<serde_json::Value> =
            tools.values().map(|tool| tool.to_json_value()).collect();
        let request = ChatCompletionRequest {
            model: &self.model,
            tools: if req_tools.is_empty() {
                None
            } else {
                Some(req_tools)
            },
            messages: messages.to_vec(),
            temperature: Some(0.67),
            parallel_tool_calls: None,
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

    /// Get a response that fits a certain schema.
    ///
    /// # Usage
    ///
    /// ```rust
    /// use openai_client::prelude::*;
    /// use serde::{Serialize, Deserialize};
    /// use schemars::JsonSchema;
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
    /// Can fail on API rejects or if the model does not respond with the correct struct.
    #[inline]
    pub async fn get_structured_response<T: DeserializeOwned + JsonSchema>(
        &self,
        messages: &[ChatCompletionMessageParam],
    ) -> Result<T> {
        let schema = schemars::schema_for!(T);
        let req = ChatCompletionRequest {
            messages: messages.to_vec(),
            model: &self.model,
            temperature: Some(0.67),
            tools: None,
            parallel_tool_calls: None,
            response_format: Some(ResponseFormat {
                kind: "json_schema".to_owned(),
                json_schema: JsonSchemaConfig {
                    name: type_name::<T>()
                        .rsplit("::")
                        .next()
                        .unwrap_or("response")
                        .to_owned(),
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
            .and_then(|response_message| response_message.content.as_deref())
            .ok_or(Error::Response)?;
        serde_json::from_str(content).map_err(Error::Deserialization)
    }

    #[inline]
    #[must_use]
    pub fn new<S1: Into<String>, S2: Into<String>>(
        url: S1,
        model: S2,
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

    /// Run the LLM in a loop until it stops calling tools and just responds.
    ///
    /// # Errors
    ///
    /// - There was no completion.
    /// - The completion did not have any messages.
    #[inline]
    pub async fn run_agent<S1: Into<String>, S2: Into<String>>(
        &self,
        system_prompt: S1,
        prompt: S2,
        tools: &ToolMap<'_>,
    ) -> Result<String> {
        let system_message = ChatCompletionMessageParam::new_system(system_prompt.into());
        let user_prompt = ChatCompletionMessageParam::new_user(prompt.into());
        let mut prompts = vec![system_message, user_prompt];
        loop {
            let completion: ChatCompletionResponse = self.get_completion(&prompts, tools).await?;
            let response = completion.first().ok_or(Error::Response)?;

            if response.has_tools() {
                prompts.push(ChatCompletionMessageParam::new_assistant_with_tools(
                    response.content.clone(),
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

    #[inline]
    #[must_use]
    pub fn set_bearer_auth<S: Into<String>>(mut self, token: S) -> Self {
        self.header_kv = Some(("Authorization".into(), token.into()));
        self
    }

    #[inline]
    #[must_use]
    pub fn set_key_pair<S1: Into<String>, S2: Into<String>>(mut self, key: S1, value: S2) -> Self {
        self.header_kv = Some((key.into(), value.into()));
        self
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionRequest<'model_info> {
    pub messages: Vec<ChatCompletionMessageParam>,
    pub model: &'model_info str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    pub temperature: Option<f32>,
    pub tools: Option<Vec<serde_json::Value>>,
}

impl<'model_name> ChatCompletionRequest<'model_name> {
    #[must_use]
    #[inline]
    pub const fn new(
        messages: Vec<ChatCompletionMessageParam>,
        model: &'model_name str,
        parallel_tool_calls: Option<bool>,
        response_format: Option<ResponseFormat>,
        temperature: Option<f32>,
        tools: Option<Vec<serde_json::Value>>,
    ) -> Self {
        Self {
            messages,
            model,
            parallel_tool_calls,
            response_format,
            temperature,
            tools,
        }
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponseFormat {
    /// Schema that the response should abide by.
    json_schema: JsonSchemaConfig,
    #[serde(rename = "type")]
    /// What kind of schema this is (almost 100% of the cases `"json_schema"`).
    kind: String,
}

#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JsonSchemaConfig {
    /// Chosen schema name.
    name: String,
    /// The schema to abide by.
    schema: serde_json::Value,
    /// If the API should return an error if an invalid schema was provided.
    strict: bool,
}

#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionMessageParam {
    pub content: Option<String>,
    pub role: AIChatRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ChatCompletionMessageParam {
    #[inline]
    #[must_use]
    pub fn new<S: Into<String>>(content: S, role: AIChatRole) -> Self {
        Self {
            content: Some(content.into()),
            role,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[inline]
    #[must_use]
    pub fn new_assistant<S: Into<String>>(content: S) -> Self {
        Self::new(content, AIChatRole::Assistant)
    }

    #[inline]
    #[must_use]
    pub const fn new_assistant_with_tools(
        content: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
    ) -> Self {
        Self {
            content,
            role: AIChatRole::Assistant,
            tool_calls,
            tool_call_id: None,
        }
    }

    #[inline]
    #[must_use]
    pub fn new_system<S: Into<String>>(content: S) -> Self {
        Self::new(content, AIChatRole::System)
    }

    #[inline]
    #[must_use]
    pub fn new_tool<S: Into<String>>(content: S, tool_call_id: &str) -> Self {
        Self {
            content: Some(content.into()),
            role: AIChatRole::Tool,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.to_owned()),
        }
    }

    #[inline]
    #[must_use]
    pub fn new_user<S: Into<String>>(content: S) -> Self {
        Self::new(content, AIChatRole::User)
    }
}

impl Default for ChatCompletionMessageParam {
    #[inline]
    fn default() -> Self {
        Self {
            content: None,
            role: AIChatRole::User,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

/// Utility method to generate a quick start of LLM turn.
#[inline]
pub fn new_system_user_turn<S1: Into<String>, S2: Into<String>>(
    system_prompt: S1,
    user_prompt: S2,
) -> Vec<ChatCompletionMessageParam> {
    vec![
        ChatCompletionMessageParam::new_system(system_prompt),
        ChatCompletionMessageParam::new_user(user_prompt),
    ]
}

/// Utility method to generate a quick user completion message parameter without developing carpal tunnel.
#[inline]
pub fn user_message<S: Into<String>>(content: S) -> ChatCompletionMessageParam {
    ChatCompletionMessageParam::new_user(content)
}

#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AIChatRole {
    #[serde(rename = "assistant")]
    Assistant,
    Custom(String),
    #[serde(rename = "system")]
    System,
    #[serde(rename = "tool")]
    Tool,
    #[serde(rename = "user")]
    User,
}

#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub choices: Vec<ChatCompletionResponseChoice>,
}

impl ChatCompletionResponse {
    #[inline]
    #[must_use]
    /// Get the first response from the model.
    pub fn first(&self) -> Option<&ChatCompletionResponseMessage> {
        self.choices.first().map(|choice| &choice.message)
    }

    #[inline]
    #[must_use]
    /// This method is currently stale as this is only usable when we configure the model to send more than a single response.
    pub fn get(&self, n: usize) -> Option<&ChatCompletionResponseMessage> {
        self.choices.get(n).map(|choice| &choice.message)
    }
}

impl From<Vec<ChatCompletionResponseChoice>> for ChatCompletionResponse {
    #[inline]
    fn from(choices: Vec<ChatCompletionResponseChoice>) -> Self {
        Self { choices }
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponseChoice {
    pub message: ChatCompletionResponseMessage,
}

#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponseMessage {
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub role: AIChatRole,
    pub tool_calls: Vec<ToolInvocation>,
}

impl ChatCompletionResponseMessage {
    #[inline]
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

    #[must_use]
    #[inline]
    pub const fn has_tools(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolInvocation {
    #[serde(rename = "type")]
    pub ftype: String,
    pub function: ToolInvocationFunction,
    pub id: String,
}

#[inline]
pub async fn async_sleep(duration: Duration) {
    #[cfg(feature = "runtime_agnostic")]
    futures_timer::Delay::new(duration).await;
    #[cfg(feature = "tokio_runtime")]
    sleep(duration).await;
}

impl ToolInvocation {
    #[inline]
    pub async fn call(&self, tools: &ToolMap<'_>) -> String {
        let Some(tool) = tools.get(&self.function.name) else {
            let name = self.function.name.as_str();
            return format!("{name} tool is unknown.");
        };
        async_sleep(tool.get_timeout_wait()).await;
        tool.invoke(
            &serde_json::from_str(&self.function.arguments).unwrap_or(serde_json::Value::Null),
        )
        .await
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolInvocationFunction {
    pub arguments: String,
    pub name: String,
}

#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
    pub id: String,
}

#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub arguments: String,
    pub name: String,
}

impl fmt::Display for ChatCompletionMessageParam {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "[{}: {}]",
            self.role,
            self.content.as_deref().unwrap_or("(null)")
        )
    }
}

#[expect(clippy::pattern_type_mismatch, reason = "Less readable")]
impl fmt::Display for AIChatRole {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let generated_name = match self {
            Self::Assistant => String::from("Assistant"),
            Self::Custom(role_name) => role_name.clone(),
            Self::System => String::from("System"),
            Self::Tool => String::from("Tool"),
            Self::User => String::from("User"),
        };
        writeln!(f, "{generated_name}")
    }
}

pub trait IntoPinBox<'str>: Sized + Send + ToString + 'str {
    /// Allow something to be returned as a `Pin<Box<dyn Future<Output = String>>>`.
    #[inline]
    fn into_pin_box(self) -> Pin<Box<dyn Future<Output = String> + Send + 'str>> {
        Box::pin(async move { self.to_string() })
    }
}

#[expect(clippy::missing_trait_methods, reason = "Readability")]
impl<'str, T> IntoPinBox<'str> for T where T: Sized + Send + ToString + 'str {}

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
///         vec![ToolCallArgDescriptor::string(
///             "cool_arg",
///             "this is a really cool argument that the llm knows about now",
///         )]
///     }
///
///     fn get_timeout_wait(&self) -> core::time::Duration {
///          core::time::Duration::from_millis(50)
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
    fn get_args(&self) -> Vec<ToolCallArgDescriptor>;
    #[must_use]
    fn get_description(&self) -> &'static str;

    #[must_use]
    fn get_name(&self) -> &'static str;

    #[inline]
    #[must_use]
    fn get_timeout_wait(&self) -> Duration {
        Duration::from_millis(3000)
    }

    fn invoke<'invocation>(
        &'invocation self,
        args: &'invocation serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = String> + Send + 'invocation>>;

    #[inline]
    #[must_use]
    fn to_json_value(&self) -> serde_json::Value {
        let name = self.get_name();
        let description = self.get_description();
        let args = self.get_args();
        let required = args
            .iter()
            .filter_map(|arg| arg.required.then_some(arg.name))
            .collect::<Vec<&str>>();
        let props: HashMap<String, serde_json::Value> = args
            .iter()
            .map(|arg_descriptor| {
                let this = &arg_descriptor;
                (
                    this.name.to_owned(),
                    serde_json::json!({
                        "type": this.argtype,
                        "description": this.description
                    }),
                )
            })
            .collect();
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
}

#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct ToolCallArgDescriptor {
    pub argtype: ToolCallArgType,
    pub description: &'static str,
    pub name: &'static str,
    pub required: bool,
}

#[non_exhaustive]
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub enum ToolCallArgType {
    #[serde(rename = "bool")]
    Bool,
    #[serde(rename = "number")]
    Number,
    #[default]
    #[serde(rename = "string")]
    String,
}

impl ToolCallArgDescriptor {
    #[inline]
    #[must_use]
    pub const fn bool(name: &'static str, description: &'static str) -> Self {
        Self::required(name, ToolCallArgType::Bool, description)
    }

    #[inline]
    #[must_use]
    pub const fn new(
        argtype: ToolCallArgType,
        description: &'static str,
        name: &'static str,
        required: bool,
    ) -> Self {
        Self {
            argtype,
            description,
            name,
            required,
        }
    }

    #[inline]
    #[must_use]
    pub const fn number(name: &'static str, description: &'static str) -> Self {
        Self::required(name, ToolCallArgType::Number, description)
    }

    #[inline]
    #[must_use]
    pub const fn optional(
        name: &'static str,
        argtype: ToolCallArgType,
        description: &'static str,
    ) -> Self {
        Self::new(argtype, description, name, false)
    }

    #[inline]
    #[must_use]
    pub const fn required(
        name: &'static str,
        argtype: ToolCallArgType,
        description: &'static str,
    ) -> Self {
        Self::new(argtype, description, name, true)
    }

    #[inline]
    #[must_use]
    pub const fn string(name: &'static str, description: &'static str) -> Self {
        Self::required(name, ToolCallArgType::String, description)
    }
}
