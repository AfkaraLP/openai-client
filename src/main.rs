use openai_client::prelude::*;
use serde_json::Value;

#[tokio::main]
async fn main() {
    let client = OpenAIClient::new("http://localhost:1234/v1", "qwen3.5-2b", None);
    let cat_shelter = client
        .get_structured_response::<Vec<Cat>>(&new_system_user_turn(
            "You are a cat shelter supervisor, you respond only in json. cats have age, name and breed.",
            "what cats are currently in your shelter?",
        ))
        .await
        .unwrap();
    println!(
        "Your cat shelter has:\n{}",
        cat_shelter.iter().fold(String::new(), |mut acc, cat| {
            use std::fmt::Write;
            _ = writeln!(
                &mut acc,
                "{name} is a {breed} cat, she is {age} old.",
                name = cat.name,
                breed = cat.breed,
                age = if cat.age == 1 {
                    "1 year".to_string()
                } else {
                    format!("{age} years", age = cat.age)
                }
            );
            acc
        })
    );

    let weather = client
        .run_agent(
            "You are a weather forecaster",
            "please tell me the weather in New Jersey",
            ToolMap::new().register_tool(WeatherTool),
        )
        .await
        .unwrap();
    println!("New Jersey Weather: {weather}");

    let haiku = client
        .get_completion(
            &new_system_user_turn("You are a poet.", "write a haiku about bread."),
            &ToolMap::new(),
        )
        .await
        .unwrap();
    println!("{}", haiku.first().unwrap().content.as_ref().unwrap());
}

#[derive(JsonSchema, Deserialize, Serialize)]
pub struct Cat {
    age: u8,
    name: String,
    breed: String,
}

struct WeatherTool;

impl ToolCallFn for WeatherTool {
    fn get_name(&self) -> &'static str {
        "weather_tool"
    }

    fn get_description(&self) -> &'static str {
        "gets the weather of some location"
    }

    fn invoke(
        &self,
        args: &serde_json::Value,
    ) -> std::pin::Pin<Box<dyn Future<Output = String> + Send>> {
        let Some(Value::String(location)) = args.get("location") else {
            return "please provide a location".into_pin_box();
        };
        format!("The weather in {location} is partly cloudy with 22 degrees").into_pin_box()
    }

    fn get_args(&self) -> Vec<ToolCallArgDescriptor> {
        vec![ToolCallArgDescriptor::new(
            true,
            "location",
            ToolCallArgType::String,
            "where to get the weather from",
        )]
    }
}
