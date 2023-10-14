use openai::{
    chat::{ChatCompletion, ChatCompletionMessage, ChatCompletionMessageRole},
    set_key,
};

use crate::config::Config;

enum Model {
    GPT3,
    GPT4,
}

impl Model {
    fn as_str(&self) -> &str {
        match self {
            Model::GPT3 => "gpt-3.5-turbo",
            Model::GPT4 => "gpt-4",
        }
    }
}

async fn make_request(role: &str, content: &str) -> Option<String> {
    let key = Config::load().unwrap().gpt_key?;
    set_key(key);

    let mut messages = vec![ChatCompletionMessage {
        role: ChatCompletionMessageRole::System,
        content: Some(role.to_string()),
        name: None,
        function_call: None,
    }];

    messages.push(ChatCompletionMessage {
        role: ChatCompletionMessageRole::User,
        content: Some(content.to_string()),
        name: None,
        function_call: None,
    });

    let chat_completion = ChatCompletion::builder(Model::GPT4.as_str(), messages.clone())
        .create()
        .await
        .ok()?;

    Some(
        chat_completion
            .choices
            .first()
            .unwrap()
            .message
            .clone()
            .content?
            .replace('"', ""),
    )
}

pub async fn get_context(question: &str, dependencies: &[String]) -> Option<String> {
    let mut role = r#"""
You will receive a question, which might lack some context to know what the question is about.
You should rewrite it, so that it will have enough context to answer it.

You will get the context from a list of other questions that the given questions depends on knowing the answer to.
here are the depencies: 


    """#.to_string();

    for dependency in dependencies {
        role.push_str(&format!("\n{}\n", dependency));
    }

    role = format!(
        "{}\n{}",
        role,
        r#"""
You will only use those depency-questions to figure out the context, you will assume the person who will answer the given question, already knows
the answers to those dependency-questions, so you will not re-ask anything from them. All you will do, is make sure the user understands the context.


You will be as concise as possible, giving only enough information so the user know what the question is asking. 

I will now give you the actual question to rewrite: 
    """#
    );

    make_request(role.as_str(), question).await
}

pub async fn get_flipped_question(question: &str, answer: &str) -> (String, String) {
    let role = r#"""
You are assisting in creating flashcards. 

Given a question and the answer to it, please help me come up with a flipped version 
where the answer becomes a part of the new question and the premise of the original question becomes the new answer. 
The response should be in the format "flipped question@@@flipped answer". 

For instance, if the question is "What is the capital of France?" and the answer is "Paris", 
the flipped version would be "Paris is the capital of which country?@@@France".

with that in mind, flip the following question and answer pair: 
"""#
    .to_string();

    let content = format!("q: {}\na: {}", question, answer);

    let response = make_request(role.as_str(), content.as_str()).await.unwrap();
    let (q, a) = response.split_once("@@@").unwrap();
    (q.to_string(), a.to_string())
}

pub async fn get_response(
    question: &str,
    dependencies: Vec<String>,
    dependents: &[String],
) -> Option<String> {
    let mut role = r#"""
You are assisting in creating flashcards. Your task is to provide a concise, singular fact as the backside of the card based on the frontside question.

To clarify:
- Your answer should be one short sentence.
- Focus only on the most important and central fact related to the question.
- Avoid dates, locations, or additional details unless they are the main focus of the question.

For example:
Question: "Who wrote 'Romeo and Juliet'?"
Correct Answer: "William Shakespeare."

This is crucial, because i need to be able to judge if i remembered the card, if there's any additional info, it gets harder to judge if i 
remembered 'enough' for it to count.

unless, of course, the question has two question marks in the end of it, then you should make it a bit more elaborate ^^

"""#.to_string();

    if !dependencies.is_empty() {
        role.push_str(
            "\nFor added context, the main question is related to the following questions:",
        );
        for dep in &dependencies {
            role.push_str(&format!("\n- {}", dep));
        }
    }

    if !dependents.is_empty() {
        role.push_str("\nAdditionally, the following questions are related to the main question:");
        for dep in dependents {
            role.push_str(&format!("\n- {}", dep));
        }
    }

    role.push_str(
        "\n\nNow, with this guidance and context in mind, please answer the following question:",
    );

    make_request(role.as_str(), question).await
}
