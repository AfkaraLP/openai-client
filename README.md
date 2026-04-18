# OpenAI Client

![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)

A simple Rust client for OpenAI-compatible APIs, with easy tool calling and agent building. I made this because tool calling with some generic OpenAI compatible endpoint like LM Studio was just so annoying with any Rust crate. So I built this as part of some project I was working on, and since I want to reuse it in other projects, I'm making it a crate on GitHub.

Nothing fancy, will probably not be well maintained or work for your use-case.

## Installation

Run the following command to add this crate as a Git dependency:

```bash
cargo add --git https://github.com/AfkaraLP/openai-client.git
```

## Features

This crate supports the following feature flags:

- `tokio_runtime` (enabled by default): Uses the Tokio runtime for asynchronous operations.
- `runtime_agnostic`: Uses futures-timer for runtime-agnostic asynchronous operations. Disable the default features if you want to use this instead.

## Example: Quick Agent Loop

Here's how to run a simple agent loop with tool calling:

```rust
use openai_client::prelude::*;
use async_trait::async_trait;

#[derive(Clone)]
struct EchoTool;

#[async_trait]
impl ToolCallFn for EchoTool {
    fn get_name(&self) -> &'static str {
        "echo"
    }

    fn get_description(&self) -> &'static str {
        "Echoes the input message"
    }

    async fn invoke(&self, args: &serde_json::Value) -> String {
        let message = args.get("message").and_then(|v| v.as_str()).unwrap_or("hello");
        println!("The model said: {message}");
        format!("Echoed: {message}")
    }

    fn get_args(&self) -> Vec<ToolCallArgDescriptor> {
        vec![ToolCallArgDescriptor::new(
            true,
            "message",
            ToolCallArgType::String,
            "The message to echo",
        )]
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OpenAIClient::new(
        "http://localhost:1234/v1".into(),
        "your-model".into(),
        None,
    );

    let tools = ToolMap::new().register_tool(EchoTool);

    let response = client.run_agent(
        &new_system_user_turn(
            "You are a helpful assistant with access to tools.",
            "Echo 'Hello World'"
        ),
        tools,
    ).await?;

    println!("Response: {}", response);
    Ok(())
}
```
