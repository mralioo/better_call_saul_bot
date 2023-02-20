# better_call_saul_bot
Chatbot with Rasa for legal counsel. It uses LLM  trained on a large corpus of legal documents. 
## Workflow 
the steps to train LLM on new legal data and deploy it in a chatbot with Rasa framework:

* Prepare your legal data: Gather a dataset of legal documents or questions that you want to use to train the LLM model. The data should be relevant to the legal domain and cover a range of topics and concepts.

* Preprocess the legal data: Clean and preprocess the legal data by removing any irrelevant information or formatting issues. Convert the legal data into a suitable format for training the LLM model.

* Train the LLM model: Use a pre-existing LLM model, such as one from Hugging Face's Transformer library, and fine-tune it on your legal dataset. You can use a machine learning framework like PyTorch or TensorFlow to train the model.

* Test the LLM model: Test the LLM model on a held-out set of legal data to evaluate its performance and accuracy. Iterate and fine-tune the model as necessary.

* Integrate the LLM model into Rasa chatbot: Integrate the trained LLM model into a Rasa chatbot by creating a custom action that utilizes the LLM model to generate legal responses to user input.

* Deploy the Rasa chatbot: Deploy the Rasa chatbot to a web server or cloud service so that users can interact with it.

* Monitor and improve: Monitor the performance of the LLM model and Rasa chatbot and make improvements as necessary. Collect feedback from users to identify areas for improvement and to ensure that the chatbot is meeting their needs.

Overall, these steps will allow you to train an LLM model on new legal data and integrate it into a Rasa chatbot so that users can receive accurate and informative legal advice.

### RASA setup 
Well this will include a whole list of steps you need to accomplish.

* Create a RASA bot on your local machine.
  * You need two ports for this. One for running RASA and one for running the Actions in rasa.
  * Create the **intents**, **entities** and the **stories**.
  * Define the responses in domain file.
* Create your own frontend and add the chatbot plugin to the website or integrate it with any channel like slack/Teams etc.
  * You can connect to any available channel by simply adding the details in credentials file.
  * You might also need a database to store all its conversation history for analytics.
  * You can use the mongoDB to activate trackers by adding the details in endpoints file.
  * You also have to mention the action endpoint here in endpoint file.
* Now once you have the developed project ready on your local machine, you need to host this to the Internet. For that you need either a public IP on which you can deploy your Bot or you can create a Docker machine to deploy it on Kubernetes.
* Deploying a chatbot is like any other software deployment. You must have an environment with all required libraries and installations and then you have to migrate your code to that host.

## Resources 
* [UNDERSTANDING HOW RASA CHATBOT WORKS](https://knowledgesmack.blogspot.com/2022/09/understanding-how-rasa-works.html)
* [UNDERSTANDING RASA NLU PIPELINE](https://knowledgesmack.blogspot.com/2022/09/understanding-rasa-nlu-pipeline.html)
* [Conversational AI chatbot using Rasa NLU & Rasa Core: How Dialogue Handling with Rasa Core can use LSTM by using Supervised and Reinforcement Learning Algorithm](https://bhashkarkunal.medium.com/conversational-ai-chatbot-using-rasa-nlu-rasa-core-how-dialogue-handling-with-rasa-core-can-use-331e7024f733)
* [Applying Generative Models in Conversational AI](https://rasa.com/blog/applying-generative-models-in-conversational-ai/)