use std::{pin::Pin, sync::Arc};

use openai_client::{
    IntoPinBox, OpenAIClient, ToolCallArgDescriptor, ToolCallArgType, ToolCallFn, ToolMap, prelude,
};

struct State;
struct Client;

pub struct SpawnSubAgent<'a> {
    state: Arc<State>,
    bot: Arc<Client>,
    username: String,
    subagent_tools: Arc<ToolMap<'a>>,
}

impl<'a> SpawnSubAgent<'a> {
    fn new(
        state: Arc<State>,
        bot: Arc<Client>,
        username: String,
        subagent_tools: Arc<ToolMap<'a>>,
    ) -> Self {
        Self {
            state,
            bot,
            username,
            subagent_tools,
        }
    }
}

// example code from my private project I stole
impl<'a> ToolCallFn for SpawnSubAgent<'a> {
    fn get_name(&self) -> &'static str {
        "spawn_sub_agent"
    }

    fn get_description(&self) -> &'static str {
        "spawn an AI that can control the bot and interact with the outside world"
    }

    fn invoke<'b>(
        &'b self,
        args: &serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = String> + Send + 'b>> {
        let Some(serde_json::Value::String(task)) = args.get("task") else {
            return "Please provide a task".into_pin_box();
        };

        let task = task.clone();
        let _state = self.state.clone();
        let _bot = self.bot.clone();
        let tools = self.subagent_tools.clone();
        let agent_loop = async move {
            let client = &OpenAIClient::new("test", "test", None);
            client
                .run_agent("test_prompt", task, &tools)
                .await
                .unwrap_or(String::from("Subagent Run Failed"))
        };
        Box::pin(agent_loop)
    }

    fn get_args(&self) -> Vec<ToolCallArgDescriptor> {
        vec![ToolCallArgDescriptor::new(
            true,
            "task",
            ToolCallArgType::String,
            "the task that the agent should fulfill before stopping.",
        )]
    }

    fn get_timeout_wait(&self) -> std::time::Duration {
        std::time::Duration::from_millis(0)
    }
}

struct SubAgentTool;

impl ToolCallFn for SubAgentTool {
    fn get_name(&self) -> &'static str {
        "thing"
    }

    fn get_description(&self) -> &'static str {
        "test"
    }

    fn invoke<'a>(
        &'a self,
        _args: &serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
        "this is just a test".into_pin_box()
    }

    fn get_args(&self) -> Vec<ToolCallArgDescriptor> {
        Vec::new()
    }
}

#[tokio::test]
async fn test_subagent_flow() {
    let subagent_tool = Arc::new(ToolMap::new().register_tool(SubAgentTool));
    let main_tool = ToolMap::new().register_tool(SpawnSubAgent::new(
        State.into(),
        Client.into(),
        "Test".into(),
        subagent_tool,
    ));

    // no-op, I am not willing to spend money or compute just on testing this, if anyone has a good idea on how to test this anyway shoot me a dm on discord or something
    let client = OpenAIClient::new("test", "test", None);
    _ = client.run_agent("test", "test", &main_tool).await;
}
