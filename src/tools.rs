use std::{
    collections::{HashMap, hash_map::Values},
    time::Duration,
};

use async_trait::async_trait;

pub type BoxedTool<'tools> = Box<dyn ToolCallFn + Send + Sync + 'tools>;
#[non_exhaustive]
#[derive(Default)]
pub struct ToolMap<'tool>(HashMap<&'tool str, BoxedTool<'tool>>);

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

/// Turn anything you want into a callable tool.
///
/// # Usage
///
/// ```rust
/// use openai_client::prelude::*;
/// use async_trait::async_trait;
///
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
/// #[async_trait]
/// impl ToolCallFn for MyTool {
///     fn get_name(&self) -> &'static str {
///         "my_tool"
///     }
///
///     fn get_description(&self) -> &'static str {
///         "this tool does awesome things"
///     }
///
///     async fn invoke(&self, args: &serde_json::Value) -> String {
///         let Some(serde_json::Value::String(cool_arg)) = args.get("cool_arg") else {
///             return "failed".to_string();
///         };
///         let context = self.some_context.clone();
///         let cool_arg = cool_arg.clone();
///         foo(cool_arg, context).await
///     }
///
///     fn get_args(&self) -> Vec<ToolCallArg> {
///         vec![ToolCallArg::string(
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
#[async_trait]
pub trait ToolCallFn {
    #[must_use]
    fn get_args(&self) -> Vec<ToolCallArg>;
    #[must_use]
    fn get_description(&self) -> &'static str;

    #[must_use]
    fn get_name(&self) -> &'static str;

    #[inline]
    #[must_use]
    /// # Time to wait before calling the tool.
    ///
    /// This method is historical baggage from the usecase I first made this crate for.
    /// You will have to manually overwrite it if you don't want any delays.
    fn get_timeout_wait(&self) -> Duration {
        Duration::from_secs(3)
    }

    async fn invoke(&self, args: &serde_json::Value) -> String;

    #[inline]
    #[must_use]
    fn to_json_value(&self) -> serde_json::Value {
        let name = self.get_name();
        let description = self.get_description();
        let args = self.get_args();
        let required: Vec<String> = args
            .iter()
            .flat_map(ToolCallArg::get_required_names)
            .collect();
        let props: serde_json::Map<String, serde_json::Value> =
            args.iter().map(ToolCallArg::to_schema).collect();
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

// TODO: maybe make use of static borrows and self borrowing instead of Boxing to avoid so many unneccessary clone calls.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum ToolCallArg {
    String {
        name: String,
        description: String,
    },
    Boolean {
        name: String,
        description: String,
    },
    Number {
        name: String,
        description: String,
    },
    Array {
        name: String,
        description: String,
        items: Box<Self>,
    },
    Object {
        name: String,
        description: String,
        props: Vec<Self>,
    },
    Optional(Box<Self>),
}

impl ToolCallArg {
    #[inline]
    #[must_use]
    pub fn string(name: impl Into<String>, description: impl Into<String>) -> Self {
        let name = name.into();
        let description = description.into();
        Self::String { name, description }
    }
    #[inline]
    #[must_use]
    pub fn bool(name: impl Into<String>, description: impl Into<String>) -> Self {
        let name = name.into();
        let description = description.into();
        Self::Boolean { name, description }
    }
    #[inline]
    #[must_use]
    pub fn number(name: impl Into<String>, description: impl Into<String>) -> Self {
        let name = name.into();
        let description = description.into();
        Self::Number { name, description }
    }
    #[inline]
    #[must_use]
    pub fn array(name: impl Into<String>, description: impl Into<String>, items: Self) -> Self {
        let name = name.into();
        let description = description.into();
        let items = Box::new(items);
        Self::Array {
            name,
            description,
            items,
        }
    }
    #[inline]
    #[must_use]
    pub fn object(
        name: impl Into<String>,
        description: impl Into<String>,
        props: Vec<Self>,
    ) -> Self {
        let name = name.into();
        let description = description.into();
        Self::Object {
            name,
            description,
            props,
        }
    }

    #[inline]
    #[must_use]
    pub fn optional(arg: Self) -> Self {
        Self::Optional(Box::new(arg))
    }

    fn get_required_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        match self {
            Self::String { name, .. }
            | Self::Boolean { name, .. }
            | Self::Number { name, .. }
            | Self::Array { name, .. } => {
                names.push(name.clone());
            }
            Self::Object {
                name,
                description: _,
                props,
            } => {
                names.push(name.clone());
                for prop in props {
                    let mut req_names = prop.get_required_names();
                    names.append(&mut req_names);
                }
            }
            Self::Optional(_) => {}
        }
        names
    }
    fn to_schema(&self) -> (String, serde_json::Value) {
        match self {
            Self::String { name, description } => (
                name.to_owned(),
                serde_json::json!({"description": description, "type": "string"}),
            ),
            Self::Boolean { name, description } => (
                name.to_owned(),
                serde_json::json!({"description": description, "type": "boolean"}),
            ),
            Self::Number { name, description } => (
                name.to_owned(),
                serde_json::json!({"description": description, "type": "number"}),
            ),
            Self::Array {
                name,
                description,
                items,
            } => {
                let arg_type = items.get_type_string();
                (
                    name.to_owned(),
                    serde_json::json!( {"description": description, "type": "array", "items": {"type": arg_type}}),
                )
            }
            Self::Object {
                name,
                description,
                props,
            } => {
                let props: serde_json::Map<String, serde_json::Value> =
                    props.iter().map(Self::to_schema).collect();
                (
                    name.clone(),
                    serde_json::json!({"description": description, "type": "object", "properties": props}),
                )
            }
            Self::Optional(new_arg_types) => new_arg_types.to_schema(),
        }
    }

    fn get_type_string(&self) -> &'static str {
        match self {
            Self::String { .. } => "string",
            Self::Boolean { .. } => "boolean",
            Self::Number { .. } => "number",
            Self::Array { .. } => "array",
            Self::Object { .. } => "object",
            Self::Optional(e) => e.get_type_string(),
        }
    }
}

pub trait ToolOutput: Sized {
    fn stringify(self) -> String;
}
