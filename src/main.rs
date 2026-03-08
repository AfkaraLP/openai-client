use openai_client::prelude::*;

fn main() {
    let client = OpenAIClient::new("https://foo.com", "some/model", None);
    client.get_structured_response::<Vec<MyFoo>>(&new_system_user_turn(
        "some system prompt",
        "some user prompt",
    ));
}

#[derive(JsonSchema, Deserialize, Serialize)]
pub struct MyFoo {
    thing: String,
}
